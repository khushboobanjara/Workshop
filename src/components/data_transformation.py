import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object


logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["age", "fever"]
            ordinal_columns = ["cough"]
            onehot_columns = ["gender", "city"]

            numerical_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean"))]
            )

            # Order represents increasing cough severity.
            ordinal_pipeline = Pipeline(
                steps=[
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            categories=[
                                ["No", "Mild", "Moderate", "Strong", "Severe"]
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1
                        )
                    )
                ]
            )

            onehot_pipeline = Pipeline(
                steps=[
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical", numerical_pipeline, numerical_columns),
                    ("ordinal", ordinal_pipeline, ordinal_columns),
                    ("categorical", onehot_pipeline, onehot_columns)
                ],
                remainder="drop"
            )

            logger.info("Preprocessor object created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logger.info("Loading training and testing datasets")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Training data shape: {train_df.shape}")
            logger.info(f"Testing data shape: {test_df.shape}")

            target_column = "has_covid"

            required_columns = [
                "age", "gender", "fever", "cough", "city", target_column
            ]

            for column in required_columns:
                if column not in train_df.columns:
                    raise ValueError(
                        f"Column '{column}' missing from training data"
                    )
                if column not in test_df.columns:
                    raise ValueError(
                        f"Column '{column}' missing from testing data"
                    )

            X_train = train_df.drop(columns=[target_column])
            X_test = test_df.drop(columns=[target_column])

            y_train = train_df[target_column].map({"Yes": 1, "No": 0})
            y_test = test_df[target_column].map({"Yes": 1, "No": 0})

            if y_train.isna().any():
                raise ValueError(
                    "Invalid target values in training data. "
                    "Expected 'Yes' or 'No'."
                )
            if y_test.isna().any():
                raise ValueError(
                    "Invalid target values in testing data. "
                    "Expected 'Yes' or 'No'."
                )

            # Normalize cough values (capitalization / whitespace)
            for dataframe in [X_train, X_test]:
                dataframe["cough"] = (
                    dataframe["cough"]
                    .astype("string")
                    .str.strip()
                    .str.capitalize()
                )

            logger.info("Input features and target variables separated")

            preprocessor = self.get_data_transformer_object()

            logger.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            X_train_transformed = np.asarray(X_train_transformed, dtype=float)
            X_test_transformed = np.asarray(X_test_transformed, dtype=float)

            train_array = np.c_[X_train_transformed, y_train.to_numpy()]
            test_array = np.c_[X_test_transformed, y_test.to_numpy()]

            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.preprocessor_obj_file_path
                ),
                exist_ok=True
            )

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logger.info("Preprocessor saved successfully")
            logger.info(f"Transformed training data shape: {train_array.shape}")
            logger.info(f"Transformed testing data shape: {test_array.shape}")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logger.exception("Error occurred during data transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()

    train_array, test_array, preprocessor_path = (
        obj.initiate_data_transformation(
            "artifacts/train.csv",
            "artifacts/test.csv"
        )
    )

    print("Data transformation completed successfully.")
    print("Training array shape:", train_array.shape)
    print("Testing array shape:", test_array.shape)
    print("Preprocessor saved at:", preprocessor_path)