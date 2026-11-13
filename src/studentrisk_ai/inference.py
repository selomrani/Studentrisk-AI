"""Inference preprocessing for the registered StudentRisk model."""

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from .data_utils import (
    OutlierCapper,
    cap_outliers,
    drop_unutilized_columns,
    fill_missing_values,
    fix_social_media,
    split_data,
)


RISK_LABELS = ("High", "Low", "Medium")
TARGET_COLUMN = "Mental_Health_Risk"


def load_champion_model(
    tracking_uri: str = "http://localhost:5000",
    model_name: str = "StudentRisk",
    model_alias: str = "champion",
):
    """Load the registered champion through MLflow's model URI."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")


@dataclass
class StudentRiskPreprocessor:
    """Fit the same transformations used by the training DAG."""

    reference_data: pd.DataFrame

    def __post_init__(self) -> None:
        if TARGET_COLUMN not in self.reference_data.columns:
            raise ValueError(f"Reference data is missing {TARGET_COLUMN!r}")

        train_df, _ = split_data(self.reference_data)
        train_df, _ = fill_missing_values(train_df)
        train_df, _ = fix_social_media(train_df)
        train_df = drop_unutilized_columns(train_df)

        self._training_columns = list(train_df.columns)
        self._outlier_capper = OutlierCapper().fit(train_df)
        train_df = self._outlier_capper.transform(train_df)
        self._categorical_columns = list(
            train_df.select_dtypes(include=["object"]).columns.drop(
                TARGET_COLUMN, errors="ignore"
            )
        )
        encoded = pd.get_dummies(
            train_df,
            columns=self._categorical_columns,
            drop_first=True,
        )
        self._feature_columns = list(encoded.drop(columns=[TARGET_COLUMN]).columns)
        self._numeric_columns = list(
            encoded.drop(columns=[TARGET_COLUMN])
            .select_dtypes(include=[np.number])
            .columns
        )
        self._minimums = encoded[self._numeric_columns].min()
        self._maximums = encoded[self._numeric_columns].max()
        self._fill_values = {}
        for column in self._training_columns:
            values = train_df[column]
            if values.dtype == "object":
                modes = values.mode()
                self._fill_values[column] = (
                    modes.iloc[0] if not modes.empty else "Unknown"
                )
            else:
                self._fill_values[column] = values.median()

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        """Transform raw student records into model-ready feature columns."""
        missing = set(self._training_columns) - set(records.columns) - {TARGET_COLUMN}
        if missing:
            raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")

        frame = records.copy()
        frame = frame.drop(columns=[TARGET_COLUMN], errors="ignore")
        for column in self._training_columns:
            if column == TARGET_COLUMN or column not in frame.columns:
                continue
            frame[column] = frame[column].fillna(self._fill_values[column])
        frame, _ = fix_social_media(frame)
        frame = drop_unutilized_columns(frame)
        frame = self._outlier_capper.transform(frame)
        encoded = pd.get_dummies(
            frame,
            columns=self._categorical_columns,
            drop_first=True,
        )
        encoded = encoded.reindex(columns=self._feature_columns, fill_value=False)
        for column in self._numeric_columns:
            minimum = self._minimums[column]
            maximum = self._maximums[column]
            if maximum != minimum:
                encoded[column] = (encoded[column] - minimum) / (maximum - minimum)
        return encoded.astype(float)
