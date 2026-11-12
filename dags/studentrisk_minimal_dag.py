"""A small Airflow learning DAG for the StudentRisk project.

This DAG intentionally does only two useful things:
1. Validate that the raw CSV exists and has the expected target column.
2. Run the existing preprocessing function and print the resulting shapes.

Training, MLflow registration, and champion promotion should be added only after
the notebook training code has been moved into an importable Python module.
"""

# Import Path so we can work with filesystem paths clearly.
from pathlib import Path

# Import pandas to read only the CSV header during validation.
import pandas as pd

# Import pendulum's datetime helper because Airflow uses timezone-aware dates.
from pendulum import datetime

# Import the decorator used to define a DAG.
from airflow.decorators import dag, task

# Import the project's existing preprocessing function.
from src.studentrisk_ai.data_utils import process_data


# This is the path to the CSV from inside the Airflow container.
# docker-compose.yml mounts the repository's ./data directory at /opt/airflow/data.
RAW_CSV = Path("/opt/airflow/data/raw/student_mental_health.csv")


# @dag tells Airflow that this function describes one workflow.
@dag(
    # This is the unique name shown in the Airflow web UI.
    dag_id="studentrisk_minimal_pipeline",
    # None means the DAG runs only when you trigger it manually.
    schedule=None,
    # Airflow uses this as the beginning of the scheduling period.
    start_date=datetime(2026, 1, 1),
    # False prevents Airflow from creating old, missed runs automatically.
    catchup=False,
    # Keep the DAG paused until you explicitly unpause or trigger it in the UI.
    is_paused_upon_creation=True,
)
def studentrisk_minimal_pipeline():
    """Define the tasks and their dependency order."""

    # @task turns this Python function into an Airflow task.
    @task
    def validate_raw_csv() -> str:
        """Check that the input file exists and has the required target."""

        # Stop the DAG clearly if the mounted input file is missing.
        if not RAW_CSV.is_file():
            raise FileNotFoundError(f"Raw CSV was not found: {RAW_CSV}")

        # Read only the header so validation does not load the complete dataset yet.
        header = pd.read_csv(RAW_CSV, nrows=0).columns

        # Fail before preprocessing if the supervised-learning target is absent.
        if "Mental_Health_Risk" not in header:
            raise ValueError("CSV is missing the 'Mental_Health_Risk' target column")

        # Returning a small string is safe; Airflow stores task results as metadata.
        print(f"Validated input: {RAW_CSV}")
        return str(RAW_CSV)

    # @task creates the second task in the workflow.
    @task
    def preprocess_data(csv_path: str) -> dict[str, tuple[int, ...]]:
        """Run the project's existing train/test preprocessing function."""

        # process_data returns X_train, y_train, X_test, and y_test in that order.
        X_train, y_train, X_test, y_test = process_data(csv_path)

        # Print useful information in the task log for learning and debugging.
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Return only compact metadata, not the complete dataframes through XCom.
        return {
            "X_train": X_train.shape,
            "y_train": y_train.shape,
            "X_test": X_test.shape,
            "y_test": y_test.shape,
        }

    # Calling a task function builds a dependency and returns an XCom reference.
    validated_csv = validate_raw_csv()

    # Passing the first task's output makes preprocessing wait for validation.
    preprocess_data(validated_csv)


# Calling the DAG factory registers this DAG object for Airflow to discover.
studentrisk_minimal_pipeline()
