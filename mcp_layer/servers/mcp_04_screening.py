"""MCP 04 - Health Screening.

Wraps the project's existing PredictPipeline. The output is always labelled a SCREENING result:
it is never a diagnosis and the response says so. Nothing is stored yet - saving and reading
screening history (screening.read.own) needs a database table, which is a later migration.
"""
from ..gateway import current_principal
from ..registry import MCPServerSpec
from ..schemas import Ownership, Role, ToolKind, ToolMeta
from ._common import BadInput, clean_text, ok, rules

ROLES = frozenset({Role.USER, Role.ADMIN, Role.SUPER_ADMIN})
SOURCE_MODEL = "ml_model"
DISCLAIMER = ("This is an automated screening result, not a medical diagnosis. "
              "Please consult a doctor to confirm or rule out any condition.")


def _number(value, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (low <= float(value) <= high):
        raise BadInput(f"{label} must be a number between {low:g} and {high:g}.")
    return float(value)


def register(spec: MCPServerSpec) -> None:
    @spec.tool(ToolMeta(
        name="run_screening", allowed_roles=ROLES, permission="screening.create.own",
        ownership=Ownership.CURRENT_USER_ONLY, kind=ToolKind.READ, requires_verified_data=True,
        description="Run the screening model on age, gender, fever, cough and city. Returns a screening "
                    "result (not a diagnosis)."))
    @rules
    def run_screening(age: int, gender: str, fever: float, cough: str, city: str) -> dict:
        current_principal()   # must be inside an authenticated gateway call
        age_value = int(_number(age, "Age", 0, 120))
        fever_value = _number(fever, "Fever", 30, 115)   # wide enough for either Celsius or Fahrenheit
        gender_value = clean_text(gender, "Gender", 1, 30)
        cough_value = clean_text(cough, "Cough", 1, 30)
        city_value = clean_text(city, "City", 1, 60)

        from src.pipeline.predict_pipeline import CustomData, PredictPipeline   # heavy imports: only when used

        features = CustomData(age_value, gender_value, fever_value, cough_value, city_value).get_data_as_dataframe()
        try:
            result, confidence = PredictPipeline().predict(features)
        except ValueError as exc:   # e.g. "'X' is not a supported value for city. Supported values: ..."
            raise BadInput(str(exc)[:300]) from None
        return ok({
            "result_type": "screening",
            "screening_result": result,
            "confidence_percent": confidence,
            "is_diagnosis": False,
            "disclaimer": DISCLAIMER,
        }, source=SOURCE_MODEL)
