"""Feature Engineering: формування офлайн-сховища ознак (feature_store) у DWH.

Три групи ознак:
1) feature_store.daily_sales   - часовий ряд щоденної виручки з лаговими/календарними ознаками;
2) feature_store.customer_rfm  - профіль покупця (RFM, сервіс, поведінка);
3) feature_store.order_service - ознаки замовлення для класифікації ризику незадоволеності.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from config import PG_DSN

LAGS = [1, 2, 3, 7, 14, 21]
TS_START, TS_END = "2017-01-01", "2018-08-15"   # стабільний період: після 15.08.2018 вивантаження неповне
HOLIDAYS = pd.to_datetime(["2017-11-24", "2017-11-25", "2017-11-26", "2017-11-27",  # Black Friday / Cyber Monday
                           "2017-12-24", "2017-12-25", "2017-12-31", "2018-01-01"])
TS_FEATURES = [f"lag_{l}" for l in LAGS] + ["roll_mean_7", "roll_mean_28", "roll_std_7",
                                             "dow", "is_weekend", "day_of_month", "month", "is_holiday"]


def make_ts_features(revenue: pd.Series) -> pd.DataFrame:
    """Ознаки часового ряду. Використовуються лише значення до дати прогнозу (без витоку даних)."""
    df = pd.DataFrame({"revenue": revenue.astype(float)})
    for lag in LAGS:
        df[f"lag_{lag}"] = df["revenue"].shift(lag)
    base = df["revenue"].shift(1)
    df["roll_mean_7"] = base.rolling(7).mean()
    df["roll_mean_28"] = base.rolling(28).mean()
    df["roll_std_7"] = base.rolling(7).std()
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_holiday"] = df.index.isin(HOLIDAYS).astype(int)
    return df


def daily_sales_features(eng) -> pd.DataFrame:
    s = pd.read_sql("select full_date, revenue from mart.sales_daily", eng, parse_dates=["full_date"])
    s = s.set_index("full_date").asfreq("D").fillna(0).loc[TS_START:TS_END]
    df = make_ts_features(s["revenue"]).dropna()
    df.index.name = "full_date"
    return df.reset_index()


def customer_features(eng) -> pd.DataFrame:
    q = """
    select c.customer_unique_id, c.state,
           max(d.full_date)                         as last_purchase,
           count(distinct f.order_id)               as frequency,
           sum(f.item_total)                        as monetary,
           avg(f.review_score)                      as avg_review,
           avg(f.delivery_days)                     as avg_delivery_days,
           avg(f.is_late::int)                      as late_share,
           count(distinct p.category_en)            as n_categories,
           avg(f.installments)                      as avg_installments,
           avg((f.payment_type = 'credit_card')::int) as credit_card_share
    from dw.fact_sales f
    join dw.dim_customer c on c.customer_key = f.customer_key
    join dw.dim_date d on d.date_key = f.date_key
    join dw.dim_product p on p.product_key = f.product_key
    join dw.dim_order_status s on s.status_key = f.status_key
    where s.is_revenue
    group by c.customer_unique_id, c.state"""
    df = pd.read_sql(q, eng, parse_dates=["last_purchase"])
    ref = df["last_purchase"].max() + pd.Timedelta(days=1)
    df["recency_days"] = (ref - df["last_purchase"]).dt.days
    for c in ["avg_review", "avg_delivery_days", "late_share", "avg_installments"]:
        df[c] = df[c].astype(float).fillna(df[c].astype(float).median())
    df["monetary"] = df["monetary"].astype(float)
    return df.drop(columns=["last_purchase"])


def order_service_features(eng) -> pd.DataFrame:
    q = """
    select f.order_id,
           min(d.full_date)                         as purchase_date,
           count(*)                                 as n_items,
           count(distinct f.seller_key)             as n_sellers,
           sum(f.price)                             as order_value,
           sum(f.freight_value)                     as freight_value,
           max(f.delivery_days)                     as delivery_days,
           max(f.delay_days)                        as delay_days,
           bool_or(f.is_late)::int                  as is_late,
           max(f.installments)                      as installments,
           max(f.payment_type)                      as payment_type,
           max(p.category_en)                       as main_category,
           bool_and(sl.state = c.state)::int        as same_state,
           max(c.state)                             as customer_state,
           avg(p.photos_qty)                        as photos_qty,
           max(f.review_score)                      as review_score
    from dw.fact_sales f
    join dw.dim_date d on d.date_key = f.date_key
    join dw.dim_product p on p.product_key = f.product_key
    join dw.dim_seller sl on sl.seller_key = f.seller_key
    join dw.dim_customer c on c.customer_key = f.customer_key
    join dw.dim_order_status s on s.status_key = f.status_key
    where s.status_name = 'delivered' and f.review_score is not null and f.delivery_days is not null
    group by f.order_id"""
    df = pd.read_sql(q, eng, parse_dates=["purchase_date"])
    df["freight_ratio"] = df["freight_value"] / df["order_value"].clip(lower=1)
    df["dissatisfied"] = (df["review_score"] <= 2).astype(int)   # цільова змінна
    return df


def build_feature_store() -> dict:
    eng = create_engine(PG_DSN)
    out = {}
    with eng.begin() as conn:
        for name, fn in [("daily_sales", daily_sales_features), ("customer_rfm", customer_features),
                         ("order_service", order_service_features)]:
            df = fn(conn)
            df.to_sql(name, conn, schema="feature_store", if_exists="replace", index=False, chunksize=10000)
            out[name] = df.shape
    return out


def load_features(name: str) -> pd.DataFrame:
    eng = create_engine(PG_DSN)
    return pd.read_sql(text(f"select * from feature_store.{name}"), eng)


if __name__ == "__main__":
    print(build_feature_store())
