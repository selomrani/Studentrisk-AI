from datetime import datetime

from airflow.decorators import dag
from airflow.operators.python import PythonOperator

from src.studentrisk_ai.data_utils import process_data


@dag(
    dag_id="process_data",
    schedule_interval=None,
    start_date=datetime(
        datetime.today().year, datetime.today().month, datetime.today().day
    ),
    catchup=False,
)
def process_data_dag():
    def run_process_data():
        X_train, y_train, X_test, y_test = process_data(
            "/opt/airflow/data/raw/student_mental_health.csv"
        )
        print(
            "Processed data shapes: "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_test={X_test.shape}, y_test={y_test.shape}"
        )

    PythonOperator(
        task_id="process_data_task",
        python_callable=run_process_data,
        do_xcom_push=False,
    )


process_data_dag()