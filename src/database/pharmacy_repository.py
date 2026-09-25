from src.database.database import get_db_connection


# =========================================================
# MEDICINES
# =========================================================

def get_all_medicines():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT
                medicine_id,
                medicine_name,
                generic_name,
                category,
                description,
                price,
                stock_quantity,
                manufacturer,
                prescription_required,
                is_available
            FROM medicines
            WHERE is_available = TRUE
            ORDER BY medicine_name
        """)

        return cursor.fetchall()

    finally:
        cursor.close()
        conn.close()


def search_medicines(search_term):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        search = f"%{search_term}%"

        cursor.execute("""
            SELECT
                medicine_id,
                medicine_name,
                generic_name,
                category,
                description,
                price,
                stock_quantity,
                manufacturer,
                prescription_required,
                is_available
            FROM medicines
            WHERE is_available = TRUE
              AND (
                    medicine_name LIKE %s
                    OR generic_name LIKE %s
                    OR category LIKE %s
                  )
            ORDER BY medicine_name
        """, (search, search, search))

        return cursor.fetchall()

    finally:
        cursor.close()
        conn.close()


def get_medicine_by_id(medicine_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT
                medicine_id,
                medicine_name,
                generic_name,
                category,
                description,
                price,
                stock_quantity,
                manufacturer,
                prescription_required,
                is_available
            FROM medicines
            WHERE medicine_id = %s
        """, (medicine_id,))

        return cursor.fetchone()

    finally:
        cursor.close()
        conn.close()


# =========================================================
# CART
# =========================================================

def add_to_cart(
    user_id,
    medicine_id,
    quantity=1
):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO pharmacy_cart
            (
                user_id,
                medicine_id,
                quantity
            )
            VALUES (%s, %s, %s)

            ON DUPLICATE KEY UPDATE
                quantity = quantity + VALUES(quantity)
        """, (
            user_id,
            medicine_id,
            quantity
        ))

        conn.commit()

    finally:
        cursor.close()
        conn.close()


def get_user_cart(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT
                pc.cart_id,
                pc.medicine_id,
                m.medicine_name,
                m.generic_name,
                m.price,
                m.stock_quantity,
                pc.quantity,
                (m.price * pc.quantity) AS item_total
            FROM pharmacy_cart pc

            JOIN medicines m
                ON pc.medicine_id = m.medicine_id

            WHERE pc.user_id = %s

            ORDER BY pc.created_at DESC
        """, (user_id,))

        return cursor.fetchall()

    finally:
        cursor.close()
        conn.close()


def remove_from_cart(
    user_id,
    medicine_id
):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            DELETE FROM pharmacy_cart

            WHERE user_id = %s
              AND medicine_id = %s
        """, (
            user_id,
            medicine_id
        ))

        conn.commit()

    finally:
        cursor.close()
        conn.close()


def update_cart_quantity(
    user_id,
    medicine_id,
    quantity
):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        if quantity <= 0:
            cursor.execute("""
                DELETE FROM pharmacy_cart

                WHERE user_id = %s
                  AND medicine_id = %s
            """, (
                user_id,
                medicine_id
            ))

        else:
            cursor.execute("""
                UPDATE pharmacy_cart

                SET quantity = %s

                WHERE user_id = %s
                  AND medicine_id = %s
            """, (
                quantity,
                user_id,
                medicine_id
            ))

        conn.commit()

    finally:
        cursor.close()
        conn.close()


def clear_cart(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            DELETE FROM pharmacy_cart
            WHERE user_id = %s
        """, (user_id,))

        conn.commit()

    finally:
        cursor.close()
        conn.close()


# =========================================================
# PHARMACY ORDERS
# =========================================================

def create_pharmacy_order(
    user_id,
    patient_name,
    phone,
    delivery_method,
    address,
    payment_method
):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:

        # -------------------------------------------------
        # GET CART
        # -------------------------------------------------

        cursor.execute("""
            SELECT
                pc.medicine_id,
                pc.quantity,
                m.medicine_name,
                m.price,
                m.stock_quantity
            FROM pharmacy_cart pc

            JOIN medicines m
                ON pc.medicine_id = m.medicine_id

            WHERE pc.user_id = %s

            FOR UPDATE
        """, (user_id,))

        cart_items = cursor.fetchall()

        if not cart_items:
            raise ValueError(
                "Your cart is empty."
            )

        # -------------------------------------------------
        # VALIDATE STOCK
        # -------------------------------------------------

        total_amount = 0

        for item in cart_items:

            if item["quantity"] > item["stock_quantity"]:
                raise ValueError(
                    f"Only "
                    f"{item['stock_quantity']} "
                    f"units of "
                    f"{item['medicine_name']} "
                    f"are available."
                )

            total_amount += (
                float(item["price"])
                * item["quantity"]
            )

        # -------------------------------------------------
        # VALIDATE DELIVERY
        # -------------------------------------------------

        if delivery_method not in [
            "CLINIC_PICKUP",
            "HOME_DELIVERY"
        ]:
            raise ValueError(
                "Invalid delivery method."
            )

        if delivery_method == "HOME_DELIVERY":

            if not address or not address.strip():
                raise ValueError(
                    "Delivery address is required."
                )

        else:
            address = None

        # -------------------------------------------------
        # VALIDATE PAYMENT
        # -------------------------------------------------

        if payment_method not in [
            "CASH",
            "ONLINE"
        ]:
            raise ValueError(
                "Invalid payment method."
            )

        # -------------------------------------------------
        # CREATE ORDER
        # -------------------------------------------------

        cursor.execute("""
            INSERT INTO pharmacy_orders
            (
                user_id,
                total_amount,
                delivery_method,
                address,
                phone,
                payment_method,
                payment_status,
                order_status
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
        """, (
            user_id,
            total_amount,
            delivery_method,
            address,
            phone,
            payment_method,
            "PENDING",
            "PLACED"
        ))

        order_id = cursor.lastrowid

        # -------------------------------------------------
        # CREATE ORDER ITEMS
        # -------------------------------------------------

        for item in cart_items:

            item_total = (
                float(item["price"])
                * item["quantity"]
            )

            cursor.execute("""
                INSERT INTO pharmacy_order_items
                (
                    order_id,
                    medicine_id,
                    quantity,
                    price,
                    item_total
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
            """, (
                order_id,
                item["medicine_id"],
                item["quantity"],
                item["price"],
                item_total
            ))

        # -------------------------------------------------
        # CASH PAYMENT
        #
        # Cash does not require online verification.
        # Reduce stock and clear cart immediately.
        # -------------------------------------------------

        if payment_method == "CASH":

            for item in cart_items:

                cursor.execute("""
                    UPDATE medicines

                    SET stock_quantity =
                        stock_quantity - %s

                    WHERE medicine_id = %s
                      AND stock_quantity >= %s
                """, (
                    item["quantity"],
                    item["medicine_id"],
                    item["quantity"]
                ))

                if cursor.rowcount != 1:
                    raise ValueError(
                        f"Insufficient stock for "
                        f"{item['medicine_name']}."
                    )

            cursor.execute("""
                DELETE FROM pharmacy_cart

                WHERE user_id = %s
            """, (user_id,))

        # -------------------------------------------------
        # ONLINE PAYMENT
        #
        # IMPORTANT:
        # Do NOT reduce stock.
        # Do NOT clear cart.
        #
        # Stock and cart are finalized only after
        # Cashfree confirms successful payment.
        # -------------------------------------------------

        conn.commit()

        return {
            "order_id": order_id,
            "total_amount": total_amount
        }

    except Exception:
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


# =========================================================
# GET ORDER
# =========================================================

# =========================================================
# GET ORDER
# =========================================================

def get_pharmacy_order(
    order_id,
    user_id
):

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:

        cursor.execute("""
            SELECT
                po.order_id,
                po.user_id,
                po.total_amount,
                po.delivery_method,
                po.address,
                po.phone,
                po.payment_method,
                po.payment_status,
                po.order_status,
                po.cashfree_order_id,
                po.cashfree_payment_session_id,
                po.created_at

            FROM pharmacy_orders po

            WHERE po.order_id = %s
              AND po.user_id = %s
        """, (
            order_id,
            user_id
        ))

        order = cursor.fetchone()

        if not order:
            return None

        cursor.execute("""
            SELECT
                poi.order_item_id,
                poi.medicine_id,
                m.medicine_name,
                poi.quantity,
                poi.price,
                poi.item_total

            FROM pharmacy_order_items poi

            JOIN medicines m
                ON poi.medicine_id = m.medicine_id

            WHERE poi.order_id = %s

            ORDER BY poi.order_item_id
        """, (order_id,))

        order["items"] = cursor.fetchall()

        return order

    finally:

        cursor.close()
        conn.close()

# =========================================================
# CASHFREE PHARMACY ORDERS
# =========================================================

def save_pharmacy_cashfree_order(
    order_id,
    cashfree_order_id,
    payment_session_id
):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            UPDATE pharmacy_orders

            SET
                cashfree_order_id = %s,
                cashfree_payment_session_id = %s

            WHERE order_id = %s
        """, (
            cashfree_order_id,
            payment_session_id,
            order_id
        ))

        conn.commit()

    finally:
        cursor.close()
        conn.close()


def get_pharmacy_order_by_cashfree_order_id(
    cashfree_order_id,
    user_id
):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT
                order_id,
                user_id,
                total_amount,
                delivery_method,
                address,
                phone,
                payment_method,
                payment_status,
                order_status,
                cashfree_order_id,
                cashfree_payment_session_id
            FROM pharmacy_orders

            WHERE cashfree_order_id = %s
              AND user_id = %s
        """, (
            cashfree_order_id,
            user_id
        ))

        return cursor.fetchone()

    finally:
        cursor.close()
        conn.close()


# =========================================================
# ONLINE PAYMENT SUCCESS
# =========================================================

def update_pharmacy_payment_success(
    order_id,
    user_id
):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:

        # -------------------------------------------------
        # LOCK ORDER
        # -------------------------------------------------

        cursor.execute("""
            SELECT
                order_id,
                user_id,
                payment_method,
                payment_status,
                order_status
            FROM pharmacy_orders

            WHERE order_id = %s
              AND user_id = %s

            FOR UPDATE
        """, (
            order_id,
            user_id
        ))

        order = cursor.fetchone()

        if not order:
            raise ValueError(
                "Pharmacy order not found."
            )

        # -------------------------------------------------
        # PREVENT DUPLICATE STOCK REDUCTION
        # -------------------------------------------------

        if (
            order["payment_status"] == "PAID"
            and
            order["order_status"] == "CONFIRMED"
        ):
            conn.commit()
            return True

        # -------------------------------------------------
        # GET ORDER ITEMS
        # -------------------------------------------------

        cursor.execute("""
            SELECT
                poi.medicine_id,
                poi.quantity,
                m.medicine_name,
                m.stock_quantity
            FROM pharmacy_order_items poi

            JOIN medicines m
                ON poi.medicine_id = m.medicine_id

            WHERE poi.order_id = %s

            FOR UPDATE
        """, (order_id,))

        order_items = cursor.fetchall()

        if not order_items:
            raise ValueError(
                "No medicines found for this order."
            )

        # -------------------------------------------------
        # CHECK STOCK AGAIN
        # -------------------------------------------------

        for item in order_items:

            if item["quantity"] > item["stock_quantity"]:
                raise ValueError(
                    f"Insufficient stock for "
                    f"{item['medicine_name']}."
                )

        # -------------------------------------------------
        # REDUCE STOCK
        # -------------------------------------------------

        for item in order_items:

            cursor.execute("""
                UPDATE medicines

                SET stock_quantity =
                    stock_quantity - %s

                WHERE medicine_id = %s
                  AND stock_quantity >= %s
            """, (
                item["quantity"],
                item["medicine_id"],
                item["quantity"]
            ))

            if cursor.rowcount != 1:
                raise ValueError(
                    f"Unable to update stock for "
                    f"{item['medicine_name']}."
                )

        # -------------------------------------------------
        # CLEAR CART
        # -------------------------------------------------

        cursor.execute("""
            DELETE FROM pharmacy_cart

            WHERE user_id = %s
        """, (user_id,))

        # -------------------------------------------------
        # MARK PAYMENT SUCCESS
        # -------------------------------------------------

        cursor.execute("""
            UPDATE pharmacy_orders

            SET
                payment_status = 'PAID',
                order_status = 'CONFIRMED'

            WHERE order_id = %s
              AND user_id = %s
        """, (
            order_id,
            user_id
        ))

        conn.commit()

        return True

    except Exception:
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


# =========================================================
# ONLINE PAYMENT FAILED
# =========================================================

def update_pharmacy_payment_failed(
    order_id,
    user_id
):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            UPDATE pharmacy_orders

            SET
                payment_status = 'FAILED'

            WHERE order_id = %s
              AND user_id = %s
        """, (
            order_id,
            user_id
        ))

        conn.commit()

    finally:
        cursor.close()
        conn.close()