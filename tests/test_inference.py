import unittest
from unittest.mock import patch

import pandas as pd

from studentrisk_ai.inference import StudentRiskPreprocessor, load_champion_model


class InferencePreprocessorTests(unittest.TestCase):
    @patch("studentrisk_ai.inference.mlflow")
    def test_loads_champion_by_alias(self, mlflow):
        expected = object()
        mlflow.pyfunc.load_model.return_value = expected

        result = load_champion_model("http://mlflow:5000")

        self.assertIs(result, expected)
        mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")
        mlflow.pyfunc.load_model.assert_called_once_with(
            "models:/StudentRisk@champion"
        )

    def test_transforms_raw_record_to_training_columns(self):
        data = pd.read_csv("data/raw/student_mental_health.csv")
        preprocessor = StudentRiskPreprocessor(data)

        transformed = preprocessor.transform(data.iloc[[0]])

        self.assertEqual(transformed.shape[0], 1)
        self.assertGreater(transformed.shape[1], 0)
        self.assertTrue(transformed.index.equals(pd.RangeIndex(1)))
        self.assertTrue((transformed.dtypes == "float64").all())

    def test_rejects_missing_required_feature(self):
        data = pd.read_csv("data/raw/student_mental_health.csv")
        preprocessor = StudentRiskPreprocessor(data)

        with self.assertRaisesRegex(ValueError, "Missing input columns"):
            preprocessor.transform(data.drop(columns=["GPA"]).iloc[[0]])


if __name__ == "__main__":
    unittest.main()
