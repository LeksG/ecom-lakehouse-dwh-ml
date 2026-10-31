"""Локальний запуск повного пайплайна (еквівалент DAG Airflow) з журналом часу виконання етапів."""
import json
import time

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
from config import REPORTS

STEPS = [("bronze_ingest", ingest_to_bronze), ("silver_transform", build_silver),
         ("data_quality_gate", run_quality_checks), ("gold_load_dwh", load_gold),
         ("sales_report", build_report), ("build_feature_store", build_feature_store),
         ("train_sales_forecast", train_forecast), ("train_customer_segmentation", segmentation),
         ("train_customer_dissatisfaction", dissatisfaction), ("train_recommender", train_recommender),
         ("batch_inference", run_inference)]

if __name__ == "__main__":
    timing = {}
    for name, fn in STEPS:
        t0 = time.time()
        fn()
        timing[name] = round(time.time() - t0, 1)
        print(f"[OK] {name}: {timing[name]} c")
    (REPORTS / "pipeline_timing.json").write_text(json.dumps(timing, indent=2))
