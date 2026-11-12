"""Minimal end-to-end StudentRisk training DAG.

The DAG validates the raw CSV, preprocesses it, trains one baseline model,
and records its metrics in MLflow. It is intentionally small so each task
and dependency is easy to understand before adding more models.
"""

import os
from pathlib import Path

import pandas as pd
from pendulum import datetime
from airflow.decorators import dag, task

from src.studentrisk_ai.data_utils import process_data


# This path exists inside the Airflow container because ./data is mounted there.
RAW_CSV = Path("/opt/airflow/data/raw/student_mental_health.csv")


@dag(
    dag_id="end_to_end_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    is_paused_upon_creation=True,
)
def end_to_end_pipeline():
    @task
    def validate_raw_csv() -> str:
        """Check the input file and required target column."""
        if not RAW_CSV.is_file():
            raise FileNotFoundError(f"Raw CSV was not found: {RAW_CSV}")

        columns = pd.read_csv(RAW_CSV, nrows=0).columns
        if "Mental_Health_Risk" not in columns:
            raise ValueError("CSV is missing the 'Mental_Health_Risk' target column")

        print(f"Validated input: {RAW_CSV}")
        return str(RAW_CSV)

    @task
    def preprocess_data(file_path: str) -> str:
        """Run the existing preprocessing and log its output shapes."""
        X_train, y_train, X_test, y_test = process_data(file_path)
        print(
            "Processed data shapes: "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_test={X_test.shape}, y_test={y_test.shape}"
        )

        # XCom carries only the small path string, not the dataframes.
        return file_path

    @task
    def train_models_with_mlflow(file_path: str) -> dict[str, object]:
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier

        X_train, y_train, X_test, y_test = process_data(file_path)
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric="mlogloss",
            ),
        }

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("StudentRisk")

        results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train_encoded)
            predictions = model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test_encoded, predictions)),
                "f1_weighted": float(
                    f1_score(y_test_encoded, predictions, average="weighted")
                ),
            }

            with mlflow.start_run(run_name=model_name) as run:
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, name="model")

            results[model_name] = {
                **metrics,
                "run_id": run.info.run_id,
            }
            print(f"{model_name}: {metrics}")

        # Select the best completed run using the same metric for every model.
        champion_name = max(
            results,
            key=lambda name: results[name]["f1_weighted"],
        )
        champion = results[champion_name]
        client = mlflow.MlflowClient()
        client.set_tag(champion["run_id"], "model_role", "champion")

        # Register the winning model and give its version a stable alias.
        registered_model = mlflow.register_model(
            f"runs:/{champion['run_id']}/model",
            "StudentRisk",
        )
        client.set_registered_model_alias(
            "StudentRisk",
            "champion",
            registered_model.version,
        )

        print(
            f"Champion: {champion_name} "
            f"(weighted F1={champion['f1_weighted']:.4f}, "
            f"version={registered_model.version})"
        )
        results["champion"] = {
            "model": champion_name,
            "run_id": champion["run_id"],
            "version": int(registered_model.version),
            "f1_weighted": champion["f1_weighted"],
        }
        return results

    validated_file = validate_raw_csv()
    processed_file = preprocess_data(validated_file)
    train_models_with_mlflow(processed_file)


end_to_end_pipeline()
