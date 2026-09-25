booking_state = {
    "active": False,
    "step": None,

    "specialty": None,

    "doctor_id": None,
    "doctor_name": None,

    "availability_id": None,

    "appointment_date": None,
    "appointment_time": None,

    "patient_name": None,
    "phone": None,

    "payment_method": None
}


def reset_booking():
    booking_state.update({
        "active": False,
        "step": None,
        "specialty": None,
        "doctor_id": None,
        "doctor_name": None,
        "availability_id": None,
        "appointment_date": None,
        "appointment_time": None,
        "patient_name": None,
        "phone": None,
        "payment_method": None
    })


def start_booking():

    reset_booking()

    booking_state["active"] = True
    booking_state["step"] = "specialty"


def set_booking_value(key, value):

    if key not in booking_state:
        return False

    booking_state[key] = value
    return True


def get_booking_step():

    return booking_state["step"]


def get_booking_data():

    return {
        "specialty": booking_state["specialty"],
        "doctor_id": booking_state["doctor_id"],
        "doctor_name": booking_state["doctor_name"],
        "availability_id": booking_state["availability_id"],
        "appointment_date": booking_state["appointment_date"],
        "appointment_time": booking_state["appointment_time"],
        "patient_name": booking_state["patient_name"],
        "phone": booking_state["phone"],
        "payment_method": booking_state["payment_method"]
    }