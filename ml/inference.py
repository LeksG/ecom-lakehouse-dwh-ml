"""Інтеграція моделей з даними (batch inference pipeline).

Моделі завантажуються з MLflow Model Registry за аліасом @champion, результати записуються
у схему ml сховища: прогноз виручки на 7 днів, сегменти покупців, ризик незадоволеності
поточних замовлень та персональні рекомендації категорій.
"""
from datetime import datetime, timezone

import mlflow
import pandas as pd
from sqlalchemy import create_engine

from config import PG_DSN, MLFLOW_URI
from ml.features import TS_FEATURES, TS_END, TS_START, load_features, make_ts_features
from ml.mlflow_utils import load_sklearn
from ml.recommender import CategoryRecommender, load_history
from ml.train_customer_models import CAT, NUM, SEG_FEATURES

HORIZON = 7


def forecast_next_days(eng, model, horizon=HORIZON):
    """Рекурсивна стратегія: прогноз на день t+1 додається до ряду і використовується для t+2 ..."""
    s = pd.read_sql("select full_date, revenue from mart.sales_daily", eng, parse_dates=["full_date"])
    s = s.set_index("full_date").asfreq("D").fillna(0).loc[TS_START:TS_END]["revenue"].astype(float)
    out = []
    for _ in range(horizon):
        nxt = s.index[-1] + pd.Timedelta(days=1)
        ext = pd.concat([s, pd.Series([0.0], index=[nxt])])
        x = make_ts_features(ext).iloc[[-1]][TS_FEATURES]
        y = float(model.predict(x)[0])
        s.loc[nxt] = y
        out.append({"forecast_date": nxt.date(), "revenue_pred": round(y, 2)})
    return pd.DataFrame(out)


def run_inference() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    eng = create_engine(PG_DSN)
    ts = datetime.now(timezone.utc)
    f_model = load_sklearn("models:/ecom_sales_forecast@champion")
    seg_model = load_sklearn("models:/ecom_customer_segmentation@champion")
    dis_model = load_sklearn("models:/ecom_customer_dissatisfaction@champion")

    fc = forecast_next_days(eng, f_model)
    fc["model_version"] = mlflow.MlflowClient().get_model_version_by_alias("ecom_sales_forecast", "champion").version
    fc["scored_at"] = ts

    cust = load_features("customer_rfm")
    X = cust[SEG_FEATURES].copy()
    import numpy as np
    X["monetary"] = np.log1p(X["monetary"]); X["frequency"] = np.log1p(X["frequency"])
    X["avg_delivery_days"] = np.log1p(X["avg_delivery_days"].clip(lower=0))
    seg = cust[["customer_unique_id"]].assign(segment=seg_model.predict(X), scored_at=ts)

    orders = load_features("order_service")
    risk = orders[["order_id"]].assign(dissatisfaction_proba=dis_model.predict_proba(orders[NUM + CAT])[:, 1].round(4),
                                       scored_at=ts)

    h = load_history(eng)
    rec = CategoryRecommender().fit(h)
    hist = h.groupby(["customer_unique_id", "state"])["category_en"].apply(lambda x: sorted(set(x))).reset_index()
    hist = hist.sample(5000, random_state=7)   # демонстраційна партія
    recs = [{"customer_unique_id": r.customer_unique_id, "rank": i + 1, "category_en": c, "scored_at": ts}
            for r in hist.itertuples() for i, c in enumerate(rec.recommend(r.category_en, r.state))]
    recs = pd.DataFrame(recs)

    with eng.begin() as conn:
        for name, df in [("sales_forecast", fc), ("customer_segments", seg),
                         ("order_dissatisfaction_risk", risk), ("category_recommendations", recs)]:
            df.to_sql(name, conn, schema="ml", if_exists="replace", index=False, chunksize=10000)
    return {"sales_forecast": fc, "segments": seg["segment"].value_counts().to_dict(),
            "risk_rows": len(risk), "recommendation_rows": len(recs)}


if __name__ == "__main__":
    r = run_inference(); print(r["sales_forecast"]); print(r["segments"])
