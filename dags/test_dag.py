# test dag
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from networkx.algorithms import dag


@dag(
    dag_id="test_dag",
    start_date=datetime(2026,11,9),
    schedule_interval="@daily",
    catchup=False,
)
def test_dag():
    def print_hello():
        print("Hello World")

    hello_task = PythonOperator(
        task_id="hello_task",
        python_callable=print_hello,
    )

    hello_task

