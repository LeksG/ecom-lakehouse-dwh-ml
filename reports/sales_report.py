"""Звіт з продажів на основі дата-мартів Gold (mart.*): графіки PNG та Excel-звіт."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from config import PG_DSN, REPORTS

FIG = REPORTS / "figures"
C1, C2 = "#1f3b73", "#d9822b"


def build_report() -> dict:
    eng = create_engine(PG_DSN)
    FIG.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_sql("""select date_trunc('month', full_date)::date as month, sum(orders) orders,
                                    sum(revenue) revenue, round(sum(revenue)/sum(orders),2) avg_order_value
                             from mart.sales_daily where full_date between '2017-01-01' and '2018-08-15'
                             group by 1 order by 1""", eng, parse_dates=["month"])
    cats = pd.read_sql("""select category_en, sum(orders) orders, sum(revenue) revenue,
                                 round(avg(avg_review),2) avg_review
                          from mart.sales_category_monthly group by 1 order by revenue desc""", eng)
    states = pd.read_sql("select * from mart.sales_state order by revenue desc", eng)
    pay = pd.read_sql("""select payment_type, count(distinct order_id) orders, sum(item_total) total
                         from dw.fact_sales group by 1 order by 2 desc""", eng)
    dow = pd.read_sql("""select d.day_of_week, avg(s.revenue) avg_revenue from mart.sales_daily s
                         join dw.dim_date d on d.full_date = s.full_date
                         where s.full_date between '2017-01-01' and '2018-08-15' group by 1 order by 1""", eng)
    kpi = pd.read_sql("""select count(distinct f.order_id) orders, sum(f.price) revenue,
                                count(distinct f.customer_key) customers,
                                round(sum(f.item_total)/count(distinct f.order_id),2) aov,
                                round(avg(f.review_score),2) avg_review,
                                round(100*avg(f.is_late::int),1) late_pct
                         from dw.fact_sales f join dw.dim_order_status s on s.status_key=f.status_key
                         where s.is_revenue""", eng)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(monthly["month"], monthly["revenue"] / 1000, width=20, color=C1, label="Виручка, тис. BRL")
    ax2 = ax.twinx(); ax2.plot(monthly["month"], monthly["orders"], color=C2, marker="o", label="Замовлення")
    ax.set_ylabel("Виручка, тис. BRL"); ax2.set_ylabel("Кількість замовлень"); ax.grid(alpha=.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left"); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(FIG / "monthly_sales.png", dpi=200); plt.close(fig)

    top = cats.head(10).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].barh(top["category_en"], top["revenue"] / 1000, color=C1); ax[0].set_xlabel("Виручка, тис. BRL")
    ax[0].grid(alpha=.3, axis="x"); ax[0].tick_params(axis="y", labelsize=8)
    st = states.head(10).iloc[::-1]
    ax[1].barh(st["state"], st["revenue"] / 1000, color=C2); ax[1].set_xlabel("Виручка, тис. BRL")
    ax[1].grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(FIG / "top_categories_states.png", dpi=200); plt.close(fig)

    with pd.ExcelWriter(REPORTS / "sales_report.xlsx") as xw:
        kpi.to_excel(xw, sheet_name="KPI", index=False)
        monthly.to_excel(xw, sheet_name="Monthly", index=False)
        cats.to_excel(xw, sheet_name="Categories", index=False)
        states.to_excel(xw, sheet_name="States", index=False)
        pay.to_excel(xw, sheet_name="Payments", index=False)
        dow.to_excel(xw, sheet_name="DayOfWeek", index=False)
    return {"kpi": kpi, "monthly": monthly, "cats": cats, "states": states, "pay": pay, "dow": dow}


if __name__ == "__main__":
    r = build_report(); print(r["kpi"].T)
