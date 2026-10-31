"""Етап L (Load): завантаження Silver -> Gold (сховище PostgreSQL, схема «зірка»).

Порядок: виміри (з генерацією сурогатних ключів) -> таблиця фактів -> дата-марти.
Масове завантаження виконується через COPY для продуктивності.
"""
import io

import pandas as pd
from deltalake import DeltaTable
from sqlalchemy import create_engine, text

from config import SILVER, PG_DSN, ROOT

STATUS = {"delivered": True, "shipped": True, "invoiced": True, "processing": True,
          "approved": True, "created": False, "canceled": False, "unavailable": False}


def silver(name):
    return DeltaTable(str(SILVER / name)).to_pandas()


def copy_df(conn, df: pd.DataFrame, table: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cur = conn.connection.cursor()
    cols = ",".join(df.columns)
    cur.copy_expert(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", buf)


def run_sql_file(conn, path):
    conn.execute(text(path.read_text(encoding="utf-8")))


def load_gold() -> dict:
    eng = create_engine(PG_DSN)
    o, i, c = silver("orders"), silver("order_items"), silver("customers")
    p, s, pay, r = silver("products"), silver("sellers"), silver("payments"), silver("reviews")

    with eng.begin() as conn:
        run_sql_file(conn, ROOT / "sql" / "ddl_star_schema.sql")

        # dim_date - повний календар періоду
        dates = pd.date_range(o["order_purchase_timestamp"].min().normalize(),
                              o["order_purchase_timestamp"].max().normalize(), freq="D")
        dd = pd.DataFrame({"full_date": dates})
        dd["date_key"] = dd["full_date"].dt.strftime("%Y%m%d").astype(int)
        dd["day_of_month"] = dd["full_date"].dt.day
        dd["month"] = dd["full_date"].dt.month
        dd["month_name"] = dd["full_date"].dt.month_name()
        dd["quarter"] = dd["full_date"].dt.quarter
        dd["year"] = dd["full_date"].dt.year
        dd["day_of_week"] = dd["full_date"].dt.dayofweek
        dd["is_weekend"] = dd["day_of_week"] >= 5
        dd["full_date"] = dd["full_date"].dt.date
        copy_df(conn, dd[["date_key", "full_date", "day_of_month", "month", "month_name", "quarter",
                          "year", "day_of_week", "is_weekend"]], "dw.dim_date")

        # dim_customer - SCD тип 1 за customer_unique_id (атрибути з останнього замовлення)
        oc = o.merge(c, on="customer_id")
        oc = oc.sort_values("order_purchase_timestamp")
        first = oc.groupby("customer_unique_id")["order_purchase_timestamp"].min().dt.date
        last = oc.groupby("customer_unique_id").tail(1)
        dc = last[["customer_unique_id", "customer_city", "customer_state", "customer_zip_code_prefix"]].copy()
        dc.columns = ["customer_unique_id", "city", "state", "zip_prefix"]
        dc["first_purchase_date"] = dc["customer_unique_id"].map(first)
        copy_df(conn, dc, "dw.dim_customer")

        dp = p.rename(columns={"product_category_name": "category_pt", "product_category_name_english": "category_en",
                               "product_weight_g": "weight_g", "product_volume_cm3": "volume_cm3",
                               "product_photos_qty": "photos_qty", "product_description_length": "description_len"})
        dp["photos_qty"] = dp["photos_qty"].round().astype(int)
        dp["description_len"] = dp["description_len"].round().astype(int)
        copy_df(conn, dp[["product_id", "category_pt", "category_en", "weight_g", "volume_cm3", "photos_qty",
                          "description_len"]], "dw.dim_product")
        ds = s.rename(columns={"seller_city": "city", "seller_state": "state"})
        copy_df(conn, ds[["seller_id", "city", "state"]], "dw.dim_seller")
        st = pd.DataFrame({"status_name": list(STATUS), "is_revenue": list(STATUS.values())})
        st.insert(0, "status_key", range(1, len(st) + 1))
        copy_df(conn, st, "dw.dim_order_status")

        # Lookup сурогатних ключів
        k = lambda q: pd.read_sql(q, conn)
        kc = k("select customer_key, customer_unique_id from dw.dim_customer")
        kp = k("select product_key, product_id from dw.dim_product")
        ks = k("select seller_key, seller_id from dw.dim_seller")

        main_pay = (pay.sort_values(["order_id", "payment_value"], ascending=[True, False])
                       .groupby("order_id").first()[["payment_type", "payment_installments"]].reset_index())
        f = (i.merge(o, on="order_id").merge(c[["customer_id", "customer_unique_id"]], on="customer_id")
              .merge(kc, on="customer_unique_id").merge(kp, on="product_id").merge(ks, on="seller_id")
              .merge(st[["status_key", "status_name"]], left_on="order_status", right_on="status_name")
              .merge(main_pay, on="order_id", how="left")
              .merge(r[["order_id", "review_score"]], on="order_id", how="left"))
        f["date_key"] = f["order_purchase_timestamp"].dt.strftime("%Y%m%d").astype(int)
        f["item_total"] = (f["price"] + f["freight_value"]).round(2)
        f = f.rename(columns={"payment_installments": "installments"})
        f["installments"] = f["installments"].astype("Int64")
        f["review_score"] = f["review_score"].astype("Int64")
        for col in ["delivery_days", "delay_days"]:
            f[col] = f[col].round(2)
        cols = ["order_id", "order_item_id", "date_key", "customer_key", "product_key", "seller_key", "status_key",
                "payment_type", "installments", "price", "freight_value", "item_total", "delivery_days",
                "delay_days", "is_late", "review_score"]
        f["is_late"] = f["is_late"].astype("boolean")
        f.loc[f["delivery_days"].isna(), "is_late"] = pd.NA
        copy_df(conn, f[cols], "dw.fact_sales")
        run_sql_file(conn, ROOT / "sql" / "marts.sql")

        counts = {t: conn.execute(text(f"select count(*) from {t}")).scalar() for t in
                  ["dw.dim_date", "dw.dim_customer", "dw.dim_product", "dw.dim_seller", "dw.dim_order_status",
                   "dw.fact_sales", "mart.sales_daily", "mart.sales_category_monthly", "mart.sales_state"]}
    return counts


if __name__ == "__main__":
    print(load_gold())
