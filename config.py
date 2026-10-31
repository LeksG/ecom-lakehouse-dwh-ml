"""Глобальна конфігурація проєкту E-commerce Lakehouse + DWH + ML."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"            # джерело (CSV-вивантаження OLTP-системи)
LAKE_DIR = ROOT / "lake"                   # озеро даних (Delta Lake)
BRONZE = LAKE_DIR / "bronze"
SILVER = LAKE_DIR / "silver"
REPORTS = ROOT / "reports"
DQ_REPORT = REPORTS / "dq_report.json"

PG_DSN = os.getenv("ECOM_DWH_DSN", "postgresql+psycopg2://postgres@/ecom_dwh?host=/tmp")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACTS = ROOT / "mlruns"

SOURCES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}
