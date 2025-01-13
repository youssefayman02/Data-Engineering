from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from functions import extract_clean, transform, load_to_postgres
from fintech_dashboard import create_dashboard



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'retries': 1
}

dag = DAG(
    'fintech_etl_pipeline',
    default_args=default_args,
    description='fintech etl pipeline',
)
with DAG(
    dag_id='fintech_etl_pipeline',
    schedule_interval = '@once',
    default_args=default_args,
    tags = ['fintech-pipeline'],
) as dag:

    extract_clean_task = PythonOperator(
        task_id='extract_clean',
        python_callable=extract_clean,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_data_22_52_14669.csv',
        }
    )

    transform_task = PythonOperator(
        task_id='transform',
        python_callable=transform,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_clean.csv',
        }
    )

    load_to_db_task = PythonOperator(
        task_id='load_to_db',
        python_callable=load_to_postgres,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_transformed.csv',
        }
    )

    run_dashboard_task = PythonOperator(
        task_id='run_dashboard',
        python_callable=create_dashboard,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_transformed.csv',
        }
    )


    extract_clean_task >> transform_task >> load_to_db_task >> run_dashboard_task