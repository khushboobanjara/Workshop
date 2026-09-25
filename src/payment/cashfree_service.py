import os

from dotenv import load_dotenv

from cashfree_pg.api_client import Cashfree
from cashfree_pg.models.create_order_request import CreateOrderRequest
from cashfree_pg.models.customer_details import CustomerDetails
from cashfree_pg.models.order_meta import OrderMeta


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


CASHFREE_CLIENT_ID = os.getenv(
    "CASHFREE_CLIENT_ID"
)

CASHFREE_CLIENT_SECRET = os.getenv(
    "CASHFREE_CLIENT_SECRET"
)

CASHFREE_ENVIRONMENT = os.getenv(
    "CASHFREE_ENVIRONMENT",
    "SANDBOX"
).upper()


# =========================================================
# VALIDATE CONFIGURATION
# =========================================================

if not CASHFREE_CLIENT_ID:

    raise RuntimeError(
        "CASHFREE_CLIENT_ID is not configured in .env"
    )


if not CASHFREE_CLIENT_SECRET:

    raise RuntimeError(
        "CASHFREE_CLIENT_SECRET is not configured in .env"
    )


# =========================================================
# GET CASHFREE CLIENT
# =========================================================

def get_cashfree_client():

    if CASHFREE_ENVIRONMENT == "PRODUCTION":

        environment = Cashfree.PRODUCTION

    else:

        environment = Cashfree.SANDBOX


    return Cashfree(
        XEnvironment=environment,
        XClientId=CASHFREE_CLIENT_ID,
        XClientSecret=CASHFREE_CLIENT_SECRET
    )


# =========================================================
# NORMALIZE CUSTOMER ID
# =========================================================

def normalize_customer_id(customer_id):

    customer_id = str(
        customer_id
    ).strip()


    # Cashfree requires at least 3 characters.
    # Example:
    # 1 -> user_1

    if len(customer_id) < 3:

        customer_id = (
            f"user_{customer_id}"
        )


    return customer_id


# =========================================================
# NORMALIZE CUSTOMER PHONE
# =========================================================

def normalize_customer_phone(customer_phone):

    customer_phone = str(
        customer_phone
    ).strip()


    # Keep digits only
    customer_phone = "".join(
        digit
        for digit in customer_phone
        if digit.isdigit()
    )


    # Example:
    # 06378066951
    # becomes
    # 6378066951

    if (
        len(customer_phone) == 11
        and customer_phone.startswith("0")
    ):

        customer_phone = (
            customer_phone[1:]
        )


    # Example:
    # 916378066951
    # becomes
    # 6378066951

    elif (
        len(customer_phone) == 12
        and customer_phone.startswith("91")
    ):

        customer_phone = (
            customer_phone[2:]
        )


    # Cashfree requires 10 digit phone number

    if len(customer_phone) != 10:

        raise ValueError(
            "Customer phone number must contain "
            "exactly 10 digits."
        )


    return customer_phone


# =========================================================
# CREATE CASHFREE ORDER
# =========================================================

def create_cashfree_order(
    order_id: str,
    amount: float,
    customer_id: str,
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    return_url: str
):

    # -----------------------------------------------------
    # Normalize customer information
    # -----------------------------------------------------

    customer_id = normalize_customer_id(
        customer_id
    )


    customer_phone = normalize_customer_phone(
        customer_phone
    )


    customer_name = str(
        customer_name
    ).strip()


    customer_email = str(
        customer_email
    ).strip()


    # -----------------------------------------------------
    # Customer details
    # -----------------------------------------------------

    customer_details = CustomerDetails(

        customer_id=customer_id,

        customer_name=customer_name,

        customer_email=customer_email,

        customer_phone=customer_phone
    )


    # -----------------------------------------------------
    # Return URL
    # -----------------------------------------------------

    order_meta = OrderMeta(
        return_url=return_url
    )


    # -----------------------------------------------------
    # Create order request
    # -----------------------------------------------------

    create_order_request = CreateOrderRequest(

        order_id=str(
            order_id
        ),

        order_amount=float(
            amount
        ),

        order_currency="INR",

        customer_details=customer_details,

        order_meta=order_meta
    )


    # -----------------------------------------------------
    # Create order using Cashfree
    # -----------------------------------------------------

    try:

        cashfree = get_cashfree_client()


        api_response = cashfree.PGCreateOrder(
            create_order_request,
            None,
            None
        )


        data = api_response.data


        return {

            "order_id": data.order_id,

            "payment_session_id":
                data.payment_session_id

        }


    except Exception as e:

        print()
        print("========================================")
        print("Cashfree Create Order Error")
        print("========================================")

        print(
            "Error Type:",
            type(e).__name__
        )

        print(
            "Error:",
            e
        )

        print("========================================")
        print()

        raise


# =========================================================
# GET CASHFREE ORDER PAYMENTS
# =========================================================

def get_cashfree_order_payments(
    order_id: str
):

    try:

        cashfree = get_cashfree_client()


        api_response = cashfree.PGOrderFetchPayments(
            str(order_id),
            None,
            None
        )


        return api_response.data


    except Exception as e:

        print()
        print("========================================")
        print("Cashfree Get Payments Error")
        print("========================================")

        print(
            "Error Type:",
            type(e).__name__
        )

        print(
            "Error:",
            e
        )

        print("========================================")
        print()

        raise