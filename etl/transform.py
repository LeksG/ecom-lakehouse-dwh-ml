"""Етап T (Transform): очищення, типізація, нормалізація та дедуплікація.

Bronze (сирі рядки) -> Silver (очищені, типізовані Delta-таблиці).
Кожне правило очищення фіксується у журналі cleaning_log для звіту про якість даних.
"""
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from config import BRONZE, SILVER

cleaning_log: list[dict] = []


def _log(table, rule, affected):
    cleaning_log.append({"table": table, "rule": rule, "affected_rows": int(affected)})


def read_bronze(name: str) -> pd.DataFrame:
    df = DeltaTable(str(BRONZE / name)).to_pandas()
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def write_silver(name: str, df: pd.DataFrame):
    write_deltalake(str(SILVER / name), pa.Table.from_pandas(df, preserve_index=False),
                    mode="overwrite", schema_mode="overwrite")


def norm_text(s: pd.Series) -> pd.Series:
    """Нормалізація назв міст: нижній регістр, без діакритики та зайвих пробілів."""
    def f(x):
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return " ".join(x.lower().replace("-", " ").split())
    return s.map(f)


def to_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.replace("", np.nan), errors="coerce")


def dedup(df, keys, table):
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="last")
    _log(table, f"видалення дублікатів за ключем {keys}", before - len(df))
    return df


def clean_orders():
    df = read_bronze("orders")
    df = dedup(df, ["order_id"], "orders")
    for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
              "order_delivered_customer_date", "order_estimated_delivery_date"]:
        df[c] = to_ts(df[c])
    df["order_status"] = df["order_status"].str.strip().str.lower()
    bad = (df["order_status"] == "delivered") & df["order_delivered_customer_date"].isna()
    _log("orders", "статус delivered без дати доставки -> позначено is_inconsistent", bad.sum())
    df["is_inconsistent"] = bad
    neg = df["order_delivered_customer_date"] < df["order_purchase_timestamp"]
    _log("orders", "дата доставки раніше дати покупки -> дату доставки обнулено", neg.sum())
    df.loc[neg, "order_delivered_customer_date"] = pd.NaT
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    df["is_late"] = df["delay_days"] > 0
    write_silver("orders", df)
    return df


def clean_order_items():
    df = read_bronze("order_items")
    df = dedup(df, ["order_id", "order_item_id"], "order_items")
    df["order_item_id"] = df["order_item_id"].astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["freight_value"] = pd.to_numeric(df["freight_value"], errors="coerce")
    df["shipping_limit_date"] = to_ts(df["shipping_limit_date"])
    bad = (df["price"].isna()) | (df["price"] <= 0)
    _log("order_items", "нульова/відʼємна або нечислова ціна -> рядок вилучено", bad.sum())
    df = df[~bad]
    fneg = df["freight_value"] < 0
    df.loc[fneg, "freight_value"] = 0
    _log("order_items", "відʼємна вартість доставки -> 0", fneg.sum())
    write_silver("order_items", df)
    return df


def clean_customers():
    df = read_bronze("customers")
    df = dedup(df, ["customer_id"], "customers")
    df["customer_zip_code_prefix"] = df["customer_zip_code_prefix"].str.zfill(5)
    before = df["customer_city"].copy()
    df["customer_city"] = norm_text(df["customer_city"])
    _log("customers", "нормалізація назв міст (регістр, діакритика)", (before != df["customer_city"]).sum())
    df["customer_state"] = df["customer_state"].str.upper().str.strip()
    write_silver("customers", df)
    return df


def clean_sellers():
    df = read_bronze("sellers")
    df = dedup(df, ["seller_id"], "sellers")
    df["seller_zip_code_prefix"] = df["seller_zip_code_prefix"].str.zfill(5)
    before = df["seller_city"].copy()
    df["seller_city"] = norm_text(df["seller_city"])
    _log("sellers", "нормалізація назв міст (регістр, діакритика)", (before != df["seller_city"]).sum())
    df["seller_state"] = df["seller_state"].str.upper().str.strip()
    write_silver("sellers", df)
    return df


def clean_products():
    df = read_bronze("products")
    tr = read_bronze("category_translation")
    df = dedup(df, ["product_id"], "products")
    df = df.rename(columns={"product_name_lenght": "product_name_length",
                            "product_description_lenght": "product_description_length"})
    miss = df["product_category_name"].eq("")
    _log("products", "пропущена категорія -> 'unknown'", miss.sum())
    df.loc[miss, "product_category_name"] = "unknown"
    df = df.merge(tr, on="product_category_name", how="left")
    no_tr = df["product_category_name_english"].isna()
    _log("products", "категорія без перекладу -> оригінальна назва", (no_tr & ~miss).sum())
    df["product_category_name_english"] = df["product_category_name_english"].fillna(df["product_category_name"])
    num = ["product_name_length", "product_description_length", "product_photos_qty",
           "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    for c in num:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
        n = df[c].isna().sum()
        med = df.groupby("product_category_name")[c].transform("median")
        df[c] = df[c].fillna(med).fillna(df[c].median())
        _log("products", f"пропуски {c} -> медіана категорії", n)
    df["product_volume_cm3"] = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    write_silver("products", df)
    return df


def clean_payments():
    df = read_bronze("payments")
    df = dedup(df, ["order_id", "payment_sequential"], "payments")
    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce").fillna(0)
    df["payment_installments"] = pd.to_numeric(df["payment_installments"], errors="coerce").fillna(1).astype(int)
    nd = df["payment_type"].eq("not_defined")
    _log("payments", "payment_type not_defined -> unknown", nd.sum())
    df.loc[nd, "payment_type"] = "unknown"
    z = df["payment_installments"] < 1
    df.loc[z, "payment_installments"] = 1
    _log("payments", "кількість платежів < 1 -> 1", z.sum())
    write_silver("payments", df)
    return df


def clean_reviews():
    df = read_bronze("reviews")
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["review_creation_date"] = to_ts(df["review_creation_date"])
    df["review_answer_timestamp"] = to_ts(df["review_answer_timestamp"])
    bad = ~df["review_score"].between(1, 5)
    _log("reviews", "оцінка поза діапазоном 1..5 -> вилучено", bad.sum())
    df = df[~bad]
    df = df.sort_values("review_answer_timestamp")
    df = dedup(df, ["order_id"], "reviews")  # одна (остання) оцінка на замовлення
    df["review_score"] = df["review_score"].astype(int)
    df["has_comment"] = df["review_comment_message"].str.strip().ne("")
    write_silver("reviews", df[["review_id", "order_id", "review_score", "has_comment",
                                "review_creation_date", "review_answer_timestamp"]])
    return df


def build_silver() -> list[dict]:
    cleaning_log.clear()
    for fn in [clean_orders, clean_order_items, clean_customers, clean_sellers,
               clean_products, clean_payments, clean_reviews]:
        fn()
    return list(cleaning_log)


if __name__ == "__main__":
    for r in build_silver():
        print(r)
