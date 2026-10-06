import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_object

logger = get_logger(__name__)


class CustomData:
    def __init__(self, age: int, gender: str, fever: float, cough: str, city: str):
        self.age = int(age)
        self.gender = str(gender).strip()
        self.fever = float(fever)
        self.cough = str(cough).strip()
        self.city = str(city).strip()

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame({
                "age": [self.age],
                "gender": [self.gender],
                "fever": [self.fever],
                "cough": [self.cough],
                "city": [self.city],
            })
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    @staticmethod
    def _known_categories(preprocessor) -> dict:
        """Return {column_name: [allowed values]} learned during training."""
        known = {}
        for _, transformer, cols in getattr(preprocessor, "transformers_", []):
            if isinstance(transformer, str):  # 'drop' / 'passthrough'
                continue
            if isinstance(cols, str):
                cols = [cols]

            if hasattr(transformer, "named_steps"):
                steps = list(transformer.named_steps.values())
            else:
                steps = [transformer]

            for step in steps:
                categories = getattr(step, "categories_", None)
                if categories is None:
                    continue
                for col, values in zip(cols, categories):
                    known[col] = [str(v) for v in values]
        return known

    def _validate_categories(self, preprocessor, features: pd.DataFrame):
        known = self._known_categories(preprocessor)
        for col, allowed in known.items():
            if col not in features.columns:
                continue
            bad = sorted(set(features[col].astype(str)) - set(allowed))
            if bad:
                raise ValueError(
                    f"'{bad[0]}' is not a supported value for {col}. "
                    f"Supported values: {', '.join(allowed)}."
                )

    def predict(self, features: pd.DataFrame):
        try:
            logger.info("Loading Model and Preprocessor")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Fail early with a clear message if the form sent an unseen value
            self._validate_categories(preprocessor, features)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            probability = None
            if hasattr(model, "predict_proba"):
                probability = round(max(model.predict_proba(data_scaled)[0]) * 100, 2)

            result = "Positive" if int(prediction[0]) == 1 else "Negative"
            logger.info(f"Prediction completed: {result}, confidence_score = {probability}")
            return result, probability

        except ValueError:
            # Let validation errors reach the route so the user sees the message
            raise
        except Exception as e:
            raise CustomException(e, sys)