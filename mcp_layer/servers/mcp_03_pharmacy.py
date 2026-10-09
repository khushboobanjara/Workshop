"""MCP 03 - Pharmacy.

Medicine availability comes ONLY from the medicines table. If a medicine is not in the database
the tool says so; nothing is ever guessed. Online payment stays in the existing Cashfree flow:
orders created through MCP are pay-at-clinic (CASH).
"""
from typing import Optional

from src.database import mcp_repository as repo
from src.database import pharmacy_repository as pharmacy

from ..gateway import current_principal
from ..registry import MCPServerSpec
from ..schemas import ErrorCode, Ownership, Role, ToolKind, ToolMeta
from ._common import BadInput, clean_phone, clean_text, fail, ok, ok_rows, positive_int, public_rows, rules

ROLES = frozenset({Role.USER, Role.ADMIN, Role.SUPER_ADMIN})
GATEWAY_FIELDS = ("cashfree_order_id", "cashfree_payment_session_id")
DELIVERY_METHODS = ("CLINIC_PICKUP", "HOME_DELIVERY")


def register(spec: MCPServerSpec) -> None:
    @spec.tool(ToolMeta(
        name="search_medicine", allowed_roles=ROLES, permission="pharmacy.search",
        ownership=Ownership.NONE, kind=ToolKind.READ, requires_verified_data=True,
        description="Search medicines by name, generic name or category."))
    @rules
    def search_medicine(query: str) -> dict:
        term = clean_text(query, "Search text", 2, 60)
        if any(ch in term for ch in "%_\\"):
            raise BadInput("Search text contains unsupported characters.")
        return ok_rows(pharmacy.search_medicines(term), "No matching medicine was found in our records.")

    @spec.tool(ToolMeta(
        name="check_medicine_availability", allowed_roles=ROLES, permission="pharmacy.availability.read",
        ownership=Ownership.NONE, kind=ToolKind.READ, requires_verified_data=True,
        description="Whether a medicine (by id) is in stock for a quantity."))
    @rules
    def check_medicine_availability(medicine_id: int, quantity: int = 1) -> dict:
        medicine_id = positive_int(medicine_id, "Medicine")
        quantity = positive_int(quantity, "Quantity", 1000)
        medicine = pharmacy.get_medicine_by_id(medicine_id)
        if not medicine:
            return fail(ErrorCode.NOT_FOUND, "That medicine is not in our records.")
        stock = int(medicine.get("stock_quantity") or 0)
        return ok({
            "medicine_id": medicine["medicine_id"], "medicine_name": medicine["medicine_name"],
            "price": medicine["price"], "prescription_required": medicine["prescription_required"],
            "requested_quantity": quantity,
            "available": bool(medicine["is_available"]) and stock >= quantity,
            "max_orderable": stock if medicine["is_available"] else 0,
        })

    @spec.tool(ToolMeta(
        name="find_nearby_pharmacies", allowed_roles=ROLES, permission="pharmacy.search",
        ownership=Ownership.NONE, kind=ToolKind.READ, requires_verified_data=True,
        description="Pharmacies near a latitude/longitude (Google Places). Radius 1-50 km."))
    @rules
    def find_nearby_pharmacies(latitude: float, longitude: float, radius_km: float = 10.0) -> dict:
        try:
            rows = pharmacy.get_nearby_pharmacies(latitude, longitude, radius_km)
        except ValueError as exc:   # the repository's own range checks are written for users
            raise BadInput(str(exc)[:200]) from None
        return ok_rows(rows, "No pharmacies were found nearby.", source="google_places")

    @spec.tool(ToolMeta(
        name="get_my_cart", allowed_roles=ROLES, permission="pharmacy.cart.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.READ, requires_verified_data=True,
        description="The signed-in user's medicine cart."))
    @rules
    def get_my_cart() -> dict:
        return ok_rows(pharmacy.get_user_cart(current_principal().user_id), "Your cart is empty.")

    @spec.tool(ToolMeta(
        name="add_to_my_cart", allowed_roles=ROLES, permission="pharmacy.cart.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.WRITE,
        description="Add a medicine to the signed-in user's cart (stock is checked)."))
    @rules
    def add_to_my_cart(medicine_id: int, quantity: int = 1) -> dict:
        user_id = current_principal().user_id
        medicine_id = positive_int(medicine_id, "Medicine")
        quantity = positive_int(quantity, "Quantity", 100)
        try:
            pharmacy.add_to_cart(user_id, medicine_id, quantity)
        except ValueError as exc:   # "Only 3 units ... are available." etc. - written for users
            raise BadInput(str(exc)[:200]) from None
        return ok_rows(pharmacy.get_user_cart(user_id), "Your cart is empty.")

    @spec.tool(ToolMeta(
        name="create_medicine_order", allowed_roles=ROLES, permission="pharmacy.order.create.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.WRITE,
        description="Place a pay-at-clinic order from the signed-in user's cart. "
                    "delivery_method is CLINIC_PICKUP or HOME_DELIVERY (address required)."))
    @rules
    def create_medicine_order(patient_name: str, phone: str, delivery_method: str,
                              address: Optional[str] = None) -> dict:
        user_id = current_principal().user_id
        name, number = clean_text(patient_name, "Patient name", 2, 100), clean_phone(phone)
        method = clean_text(delivery_method, "Delivery method", 3, 20).upper()
        if method not in DELIVERY_METHODS:
            raise BadInput("Delivery method must be CLINIC_PICKUP or HOME_DELIVERY.")
        where = clean_text(address, "Delivery address", 5, 300) if method == "HOME_DELIVERY" else None
        try:
            created = pharmacy.create_pharmacy_order(user_id, name, number, method, where, "CASH")
        except ValueError as exc:   # empty cart / insufficient stock - written for users
            raise BadInput(str(exc)[:200]) from None
        order = pharmacy.get_pharmacy_order(created["order_id"], user_id)
        if not order:
            return fail(ErrorCode.INTERNAL_ERROR, "The order could not be confirmed. Please check your orders.")
        return ok({k: v for k, v in order.items() if k not in GATEWAY_FIELDS}, message="Order placed.")

    @spec.tool(ToolMeta(
        name="get_my_orders", allowed_roles=ROLES, permission="pharmacy.order.read.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.READ, requires_verified_data=True,
        description="The signed-in user's own medicine orders, newest first."))
    @rules
    def get_my_orders() -> dict:
        rows = repo.list_user_pharmacy_orders(current_principal().user_id)
        return ok_rows(public_rows(rows, GATEWAY_FIELDS), "You have no medicine orders on record.")

    @spec.tool(ToolMeta(
        name="get_my_order_details", allowed_roles=ROLES, permission="pharmacy.order.read.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.READ, requires_verified_data=True,
        description="One of the signed-in user's own orders, with its items."))
    @rules
    def get_my_order_details(order_id: int) -> dict:
        order_id = positive_int(order_id, "Order")
        order = pharmacy.get_pharmacy_order(order_id, current_principal().user_id)
        if not order:
            return fail(ErrorCode.NOT_FOUND, "That order could not be found.")
        return ok({k: v for k, v in order.items() if k not in GATEWAY_FIELDS})

    @spec.tool(ToolMeta(
        name="cancel_order", allowed_roles=ROLES, permission="pharmacy.order.cancel.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.WRITE,
        description="Cancel one of the signed-in user's own orders that is still placed and unpaid."))
    @rules
    def cancel_order(order_id: int) -> dict:
        user_id = current_principal().user_id
        order_id = positive_int(order_id, "Order")
        if not pharmacy.get_pharmacy_order(order_id, user_id):
            return fail(ErrorCode.NOT_FOUND, "That order could not be found.")
        if not repo.cancel_pharmacy_order(order_id, user_id):
            return fail(ErrorCode.CONFLICT, "This order can no longer be cancelled.")
        return ok({"order_id": order_id, "order_status": "CANCELLED"}, message="Order cancelled.")
