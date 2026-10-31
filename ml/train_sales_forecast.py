"""ML-задача 1: прогнозування щоденної виручки (регресія, крок 1 день; горизонт 7 днів - рекурсивно).

Порівнюються: сезонний наївний прогноз (baseline), Ridge, RandomForest, HistGradientBoosting.
Валідація: хронологічний поділ (train до 2018-05-31, test - червень-серпень 2018) та
TimeSeriesSplit на навчальній вибірці. Найкраща модель реєструється у MLflow Model Registry.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import REPORTS
from ml.features import load_features, TS_FEATURES
from ml.mlflow_utils import setup, promote, log_sklearn, load_sklearn

warnings.filterwarnings("ignore")
FEATURES = TS_FEATURES
SPLIT_DATE = "2018-06-01"
MODEL_NAME = "ecom_sales_forecast"


def metrics(y, p):
    return {"mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mape": mean_absolute_percentage_error(y, p) * 100, "r2": r2_score(y, p)}


def train() -> pd.DataFrame:
    setup("sales_forecast")
    df = load_features("daily_sales")
    df["full_date"] = pd.to_datetime(df["full_date"])
    tr, te = df[df.full_date < SPLIT_DATE], df[df.full_date >= SPLIT_DATE]
    Xtr, ytr, Xte, yte = tr[FEATURES], tr["revenue"], te[FEATURES], te["revenue"]

    candidates = {
        "seasonal_naive": None,
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "hist_gbr": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3,
                                                  l2_regularization=1.0, random_state=42),
    }
    rows, preds = [], {}
    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({"model": name, "horizon_days": 1, "n_features": len(FEATURES),
                               "train_rows": len(tr), "test_rows": len(te), "split_date": SPLIT_DATE})
            if model is None:
                p = Xte["lag_7"].values   # прогноз = значення тиждень тому
                cv = np.nan
            else:
                cv = -cross_val_score(model, Xtr, ytr, cv=tscv, scoring="neg_mean_absolute_error").mean()
                model.fit(Xtr, ytr)
                p = model.predict(Xte)
                log_sklearn(model, Xtr, model.predict(Xtr))
            m = metrics(yte, p)
            if not np.isnan(cv):
                m["cv_mae"] = cv
            mlflow.log_metrics(m)
            rows.append({"model": name, **m, "run_id": run.info.run_id})
            preds[name] = p
    res = pd.DataFrame(rows).sort_values("mae")
    best = res[res.model != "seasonal_naive"].iloc[0]
    mv = mlflow.register_model(f"runs:/{best.run_id}/model", MODEL_NAME)
    promote(MODEL_NAME, mv.version)

    # Графік прогноз vs факт
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(te.full_date, yte, label="Фактична виручка", color="#1f3b73", lw=2)
    ax.plot(te.full_date, preds[best.model], label=f"Прогноз ({best.model})", color="#d9822b", lw=2)
    ax.plot(te.full_date, preds["seasonal_naive"], label="Сезонний наївний прогноз", color="#999", lw=1, ls="--")
    ax.set_ylabel("Виручка, BRL"); ax.grid(alpha=.3); ax.legend()
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "forecast_vs_actual.png", dpi=200); plt.close(fig)

    best_model = load_sklearn(f"models:/{MODEL_NAME}@champion")
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
    else:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(best_model, Xte, yte, n_repeats=10, random_state=42)
        imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.plot.barh(ax=ax, color="#1f3b73"); ax.set_xlabel("Важливість ознаки"); ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(REPORTS / "figures" / "forecast_importance.png", dpi=200); plt.close(fig)
    res.to_csv(REPORTS / "metrics_forecast.csv", index=False)
    return res


if __name__ == "__main__":
    print(train())
