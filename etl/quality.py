"""Перевірки якості даних (Data Quality) у стилі Great Expectations.

Кожне очікування (expectation) повертає результат success/observed; пайплайн зупиняється,
якщо критичне очікування не виконано (quality gate).
"""
import json

import pandas as pd
from deltalake import DeltaTable

from config import SILVER, DQ_REPORT


BR_STATES = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB",
             "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]


def silver(name):
    return DeltaTable(str(SILVER / name)).to_pandas()


class Suite:
    def __init__(self, table, df):
        self.table, self.df, self.results = table, df, []

    def _add(self, exp, ok, observed, critical=True):
        self.results.append({"table": self.table, "expectation": exp, "success": bool(ok),
                             "observed": observed, "critical": critical})

    def unique(self, cols):
        d = int(self.df.duplicated(subset=cols).sum())
        self._add(f"expect_compound_columns_to_be_unique{cols}", d == 0, d)

    def not_null(self, col, mostly=1.0, critical=True):
        share = float(self.df[col].notna().mean())
        self._add(f"expect_column_values_to_not_be_null({col}, mostly={mostly})", share >= mostly,
                  round(share, 4), critical)

    def between(self, col, lo, hi, mostly=1.0):
        s = self.df[col].dropna()
        share = float(s.between(lo, hi).mean()) if len(s) else 1.0
        self._add(f"expect_column_values_to_be_between({col}, {lo}, {hi})", share >= mostly, round(share, 4))

    def in_set(self, col, values):
        share = float(self.df[col].isin(values).mean())
        self._add(f"expect_column_values_to_be_in_set({col})", share == 1.0, round(share, 4))

    def fk(self, col, ref: pd.Series, ref_name, mostly=1.0):
        share = float(self.df[col].isin(set(ref)).mean())
        self._add(f"expect_foreign_key({col} -> {ref_name})", share >= mostly, round(share, 4))


def run_quality_checks() -> dict:
    o, i, c, p = silver("orders"), silver("order_items"), silver("customers"), silver("products")
    pay, r, s = silver("payments"), silver("reviews"), silver("sellers")
    suites = []
    so = Suite("orders", o); so.unique(["order_id"]); so.not_null("customer_id")
    so.not_null("order_purchase_timestamp")
    so.between("order_purchase_timestamp", pd.Timestamp("2016-01-01"), pd.Timestamp("2018-12-31"))
    so.in_set("order_status", ["delivered", "shipped", "canceled", "unavailable", "invoiced",
                               "processing", "created", "approved"])
    so.fk("customer_id", c["customer_id"], "customers.customer_id")
    so.not_null("order_delivered_customer_date", mostly=0.95, critical=False)
    suites.append(so)
    si = Suite("order_items", i); si.unique(["order_id", "order_item_id"])
    si.between("price", 0.01, 10000); si.between("freight_value", 0, 1000)
    si.fk("order_id", o["order_id"], "orders.order_id"); si.fk("product_id", p["product_id"], "products.product_id")
    si.fk("seller_id", s["seller_id"], "sellers.seller_id"); suites.append(si)
    sc = Suite("customers", c); sc.unique(["customer_id"]); sc.not_null("customer_unique_id")
    sc.in_set("customer_state", BR_STATES)
    suites.append(sc)
    sp = Suite("products", p); sp.unique(["product_id"]); sp.not_null("product_category_name_english")
    sp.between("product_weight_g", 0, 50000); suites.append(sp)
    spay = Suite("payments", pay); spay.between("payment_value", 0, 20000)
    spay.in_set("payment_type", ["credit_card", "boleto", "voucher", "debit_card", "unknown"]); suites.append(spay)
    sr = Suite("reviews", r); sr.unique(["order_id"]); sr.between("review_score", 1, 5); suites.append(sr)

    results = [x for su in suites for x in su.results]
    critical_failed = [x for x in results if x["critical"] and not x["success"]]
    report = {"total": len(results), "passed": sum(x["success"] for x in results),
              "critical_failed": len(critical_failed), "results": results}
    DQ_REPORT.parent.mkdir(exist_ok=True, parents=True)
    DQ_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if critical_failed:
        raise RuntimeError(f"Quality gate failed: {critical_failed}")
    return report


if __name__ == "__main__":
    rep = run_quality_checks()
    print(rep["passed"], "/", rep["total"])
