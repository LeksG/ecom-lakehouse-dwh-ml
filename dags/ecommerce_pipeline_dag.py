"""Apache Airflow DAG: щоденний ETL + ML пайплайн системи електронної комерції.

bronze_ingest -> silver_transform -> data_quality_gate -> gold_load_dwh -> build_feature_store
    -> [train_sales_forecast, train_customer_models, train_recommender] -> batch_inference -> sales_report
"""
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/ecom")   # шлях до проєкту в контейнері Airflow

from etl.extract import ingest_to_bronze
from etl.load_dwh import load_gold
from etl.quality import run_quality_checks
from etl.transform import build_silver
from ml.features import build_feature_store
from ml.inference import run_inference
from ml.recommender import train as train_recommender
from ml.train_customer_models import dissatisfaction, segmentation
from ml.train_sales_forecast import train as train_forecast
from reports.sales_report import build_report

default_args = {"owner": "data-engineering", "retries": 2, "retry_delay": timedelta(minutes=5),
                "do_xcom_push": False}  # без XCom

with DAG(
    dag_id="ecommerce_lakehouse_ml",
    description="E-commerce: Bronze/Silver/Gold + DWH (star) + ML (MLflow)",
    start_date=datetime(2026, 10, 1),
    schedule="0 3 * * *",          # щодня о 03:00
    catchup=False,
    default_args=default_args,
    tags=["ecommerce", "etl", "ml"],
) as dag:
    t_bronze = PythonOperator(task_id="bronze_ingest", python_callable=ingest_to_bronze)
    t_silver = PythonOperator(task_id="silver_transform", python_callable=build_silver)
    t_dq = PythonOperator(task_id="data_quality_gate", python_callable=run_quality_checks)
    t_gold = PythonOperator(task_id="gold_load_dwh", python_callable=load_gold)
    t_feat = PythonOperator(task_id="build_feature_store", python_callable=build_feature_store)
    t_fc = PythonOperator(task_id="train_sales_forecast", python_callable=train_forecast)
    t_seg = PythonOperator(task_id="train_customer_segmentation", python_callable=segmentation)
    t_dis = PythonOperator(task_id="train_customer_dissatisfaction", python_callable=dissatisfaction)
    t_rec = PythonOperator(task_id="train_recommender", python_callable=train_recommender)
    t_inf = PythonOperator(task_id="batch_inference", python_callable=run_inference)
    t_rep = PythonOperator(task_id="sales_report", python_callable=build_report)

    t_bronze >> t_silver >> t_dq >> t_gold >> t_feat >> [t_fc, t_seg, t_dis, t_rec] >> t_inf
    t_gold >> t_rep
