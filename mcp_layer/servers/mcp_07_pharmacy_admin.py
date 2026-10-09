"""MCP 07 - Pharmacy / Inventory Administration (ADMIN and SUPER_ADMIN only).

remove_medicine is the one DESTRUCTIVE tool. It needs an explicit confirm=true on top of the
pharmacy.manage permission, never deletes a medicine that appears in any order, and the registry
already guarantees a destructive tool is audited and never exposed to USER.
"""
from typing import Optional

from src.database import mcp_admin_repository as admin_repo

from ..registry import MCPServerSpec
from ..schemas import ErrorCode, Ownership, Role, ToolKind, ToolMeta
from ._common import BadInput, clean_text, fail, money, ok, ok_rows, opt_text, positive_int, rules, whole_number

ADMINS = frozenset({Role.ADMIN, Role.SUPER_ADMIN})
ORDER_STATUSES = {"CONFIRMED", "CANCELLED"}   # the only transitions an admin may request here


def _exists(medicine_id: int):
    return admin_repo.get_medicine(medicine_id)


def register(spec: MCPServerSpec) -> None:
    @spec.tool(ToolMeta(
        name="admin_list_medicines", allowed_roles=ADMINS, permission="pharmacy.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Medicines including unavailable ones, with stock. Optional search text; max 200."))
    @rules
    def admin_list_medicines(search: Optional[str] = None, only_available: bool = False) -> dict:
        term = opt_text(search, "Search text", 2, 60)
        return ok_rows(admin_repo.list_medicines(term, bool(only_available), 200), "No medicines match.")

    @spec.tool(ToolMeta(
        name="add_medicine", allowed_roles=ADMINS, permission="pharmacy.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Add a medicine to the catalogue (available immediately)."))
    @rules
    def add_medicine(medicine_name: str, generic_name: str, category: str, description: str, price: float,
                     stock_quantity: int, manufacturer: str, prescription_required: bool = False) -> dict:
        values = (
            clean_text(medicine_name, "Medicine name", 2, 120), clean_text(generic_name, "Generic name", 2, 120),
            clean_text(category, "Category", 2, 60), clean_text(description, "Description", 1, 500),
            money(price, "Price", 1_000_000), whole_number(stock_quantity, "Stock", 0, 1_000_000),
            clean_text(manufacturer, "Manufacturer", 2, 100), bool(prescription_required),
        )
        new_id = admin_repo.insert_medicine(*values)
        return ok({"medicine_id": new_id, "medicine_name": values[0]}, message="Medicine added.")

    @spec.tool(ToolMeta(
        name="update_medicine", allowed_roles=ADMINS, permission="pharmacy.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Change a medicine's details. Leave a field empty to keep it. Stock has its own tool."))
    @rules
    def update_medicine(medicine_id: int, medicine_name: Optional[str] = None, generic_name: Optional[str] = None,
                        category: Optional[str] = None, description: Optional[str] = None,
                        price: Optional[float] = None, manufacturer: Optional[str] = None,
                        prescription_required: Optional[bool] = None) -> dict:
        medicine_id = positive_int(medicine_id, "Medicine")
        fields = {}
        for column, value, label, high in (("medicine_name", medicine_name, "Medicine name", 120),
                                           ("generic_name", generic_name, "Generic name", 120),
                                           ("category", category, "Category", 60),
                                           ("description", description, "Description", 500),
                                           ("manufacturer", manufacturer, "Manufacturer", 100)):
            cleaned = opt_text(value, label, 1, high)
            if cleaned is not None:
                fields[column] = cleaned
        if price is not None:
            fields["price"] = money(price, "Price", 1_000_000)
        if prescription_required is not None:
            fields["prescription_required"] = 1 if prescription_required else 0
        if not fields:
            raise BadInput("Nothing to update.")
        if not _exists(medicine_id):
            return fail(ErrorCode.NOT_FOUND, "That medicine could not be found.")
        admin_repo.update_medicine(medicine_id, fields)
        return ok({"medicine_id": medicine_id, "updated": sorted(fields)}, message="Medicine updated.")

    @spec.tool(ToolMeta(
        name="set_medicine_stock", allowed_roles=ADMINS, permission="pharmacy.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Set the stock quantity of a medicine (0 to 1,000,000)."))
    @rules
    def set_medicine_stock(medicine_id: int, stock_quantity: int) -> dict:
        medicine_id = positive_int(medicine_id, "Medicine")
        quantity = whole_number(stock_quantity, "Stock", 0, 1_000_000)
        before = _exists(medicine_id)
        if not before:
            return fail(ErrorCode.NOT_FOUND, "That medicine could not be found.")
        previous = before["stock_quantity"]   # read before the write, not after
        admin_repo.set_medicine_stock(medicine_id, quantity)
        return ok({"medicine_id": medicine_id, "previous_stock": previous, "stock_quantity": quantity},
                  message="Stock updated.")

    @spec.tool(ToolMeta(
        name="set_medicine_availability", allowed_roles=ADMINS, permission="pharmacy.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Show or hide a medicine from patients without deleting it."))
    @rules
    def set_medicine_availability(medicine_id: int, is_available: bool) -> dict:
        medicine_id = positive_int(medicine_id, "Medicine")
        if not _exists(medicine_id):
            return fail(ErrorCode.NOT_FOUND, "That medicine could not be found.")
        admin_repo.set_medicine_available(medicine_id, bool(is_available))
        return ok({"medicine_id": medicine_id, "is_available": bool(is_available)}, message="Medicine updated.")

    @spec.tool(ToolMeta(
        name="remove_medicine", allowed_roles=ADMINS, permission="pharmacy.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.DESTRUCTIVE,
        description="PERMANENTLY delete a medicine that no order has ever used. Requires confirm=true. "
                    "Otherwise use set_medicine_availability."))
    @rules
    def remove_medicine(medicine_id: int, confirm: bool = False) -> dict:
        medicine_id = positive_int(medicine_id, "Medicine")
        if confirm is not True:
            raise BadInput("This permanently deletes the medicine. Repeat the request with confirm=true to proceed.")
        outcome = admin_repo.delete_medicine_if_unused(medicine_id)
        if outcome == "NOT_FOUND":
            return fail(ErrorCode.NOT_FOUND, "That medicine could not be found.")
        if outcome == "IN_USE":
            return fail(ErrorCode.CONFLICT, "This medicine appears in past orders and cannot be deleted. Hide it instead.")
        return ok({"medicine_id": medicine_id, "deleted": True}, message="Medicine deleted.")

    @spec.tool(ToolMeta(
        name="admin_list_pharmacy_orders", allowed_roles=ADMINS, permission="pharmacy.order.read.all",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.READ, requires_verified_data=True,
        description="Pharmacy orders from all users, newest first (max 100). Optional order_status filter."))
    @rules
    def admin_list_pharmacy_orders(order_status: Optional[str] = None) -> dict:
        wanted = None
        if order_status and order_status.strip():
            wanted = clean_text(order_status, "Status", 3, 20).upper()
        return ok_rows(admin_repo.list_pharmacy_orders(wanted, 100), "No orders match.")

    @spec.tool(ToolMeta(
        name="update_pharmacy_order_status", allowed_roles=ADMINS, permission="pharmacy.order.manage",
        ownership=Ownership.ADMINISTRATIVE, kind=ToolKind.WRITE,
        description="Move a PLACED order to CONFIRMED (cash orders) or CANCELLED (unpaid orders; cash stock is restored)."))
    @rules
    def update_pharmacy_order_status(order_id: int, new_status: str) -> dict:
        order_id = positive_int(order_id, "Order")
        target = clean_text(new_status, "Status", 3, 20).upper()
        if target not in ORDER_STATUSES:
            raise BadInput("Status must be CONFIRMED or CANCELLED.")
        outcome = admin_repo.set_pharmacy_order_status(order_id, target)
        if outcome == "NOT_FOUND":
            return fail(ErrorCode.NOT_FOUND, "That order could not be found.")
        if outcome == "NOT_ALLOWED":
            return fail(ErrorCode.CONFLICT, "That change isn't allowed for this order.")
        return ok({"order_id": order_id, "order_status": target}, message="Order updated.")
