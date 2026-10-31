"""ML-задача 2: класифікація покупців.

2а) Сегментація покупців (кластеризація KMeans за RFM-профілем, k обирається за силуетом).
2б) Класифікація покупців за ризиком незадоволеності (review_score <= 2) за ознаками сервісу
    замовлення - бінарна класифікація LogisticRegression / RandomForest / HistGradientBoosting.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import REPORTS
from ml.features import load_features
from ml.mlflow_utils import setup, promote, log_sklearn

warnings.filterwarnings("ignore")
FIG = REPORTS / "figures"
SEG_FEATURES = ["recency_days", "frequency", "monetary", "avg_review", "avg_delivery_days", "late_share"]


def segmentation():
    setup("customer_segmentation")
    df = load_features("customer_rfm")
    X = df[SEG_FEATURES].copy()
    X["monetary"] = np.log1p(X["monetary"])
    X["frequency"] = np.log1p(X["frequency"])
    X["avg_delivery_days"] = np.log1p(X["avg_delivery_days"].clip(lower=0))
    sample = X.sample(15000, random_state=42)
    sil = {}
    for k in range(3, 8):
        pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=k, n_init=10, random_state=42))
        lab = pipe.fit_predict(sample)
        sil[k] = silhouette_score(pipe[0].transform(sample), lab)
    best_k = max(sil, key=sil.get)
    with mlflow.start_run(run_name=f"kmeans_k{best_k}"):
        pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=best_k, n_init=10, random_state=42))
        df["segment"] = pipe.fit_predict(X)
        s_full = silhouette_score(pipe[0].transform(sample), pipe.predict(sample))
        mlflow.log_params({"algorithm": "KMeans", "k": best_k, "features": ",".join(SEG_FEATURES)})
        mlflow.log_metrics({"silhouette": s_full, **{f"silhouette_k{k}": v for k, v in sil.items()}})
        info = log_sklearn(pipe, X, pipe.predict(X))
    mv = mlflow.register_model(info.model_uri, "ecom_customer_segmentation")
    promote("ecom_customer_segmentation", mv.version)

    prof = df.groupby("segment").agg(customers=("customer_unique_id", "count"),
                                     recency=("recency_days", "mean"), frequency=("frequency", "mean"),
                                     monetary=("monetary", "mean"), review=("avg_review", "mean"),
                                     delivery=("avg_delivery_days", "mean"), late=("late_share", "mean")).round(2)
    prof["share_pct"] = (100 * prof["customers"] / prof["customers"].sum()).round(1)
    prof.to_csv(REPORTS / "segments_profile.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(list(sil), list(sil.values()), marker="o", color="#1f3b73")
    ax[0].set_xticks(list(sil)); ax[0].set_xlabel("Кількість кластерів k"); ax[0].set_ylabel("Коефіцієнт силуету")
    ax[0].grid(alpha=.3)
    cols = ["frequency", "monetary", "review", "delivery", "late"]
    names = ["Частота", "Сума", "Оцінка", "Доставка, дн.", "Частка\nзатримок"]
    norm = prof[cols] / prof[cols].max()
    w = 0.8 / len(norm)
    for n, (seg, row) in enumerate(norm.iterrows()):
        ax[1].bar(np.arange(len(cols)) + n * w, row.values, width=w,
                  label=f"Сегмент {seg} ({prof.loc[seg, 'share_pct']}%)")
    ax[1].set_xticks(np.arange(len(cols)) + w * (len(norm) - 1) / 2, names)
    ax[1].set_ylabel("Середнє значення / максимум"); ax[1].set_ylim(0, 1.25)
    ax[1].legend(fontsize=8, ncol=3, loc="upper center"); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "segmentation.png", dpi=200); plt.close(fig)
    return sil, prof, df[["customer_unique_id", "segment"]]


NUM = ["n_items", "n_sellers", "order_value", "freight_value", "freight_ratio", "delivery_days", "delay_days",
       "is_late", "installments", "same_state", "photos_qty"]
CAT = ["payment_type", "main_category", "customer_state"]


def dissatisfaction():
    setup("customer_dissatisfaction")
    df = load_features("order_service")
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df = df.sort_values("purchase_date")
    tr, te = df[df.purchase_date < "2018-04-01"], df[df.purchase_date >= "2018-04-01"]   # хронологічний поділ
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="unknown"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                            sparse_output=False)), CAT)])
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced_subsample",
                                                random_state=42, n_jobs=-1),
        "hist_gbc": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5,
                                                   class_weight="balanced", random_state=42),
    }
    rows, probs = [], {}
    for name, clf in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("prep", pre), ("clf", clf)])
            pipe.fit(tr[NUM + CAT], tr["dissatisfied"])
            p = pipe.predict_proba(te[NUM + CAT])[:, 1]
            # поріг, що максимізує F1 на навчальній вибірці
            ptr = pipe.predict_proba(tr[NUM + CAT])[:, 1]
            grid = np.linspace(0.2, 0.9, 71)
            thr = grid[np.argmax([f1_score(tr["dissatisfied"], ptr >= t) for t in grid])]
            yhat = (p >= thr).astype(int)
            m = {"roc_auc": roc_auc_score(te["dissatisfied"], p), "pr_auc": average_precision_score(te["dissatisfied"], p),
                 "precision": precision_score(te["dissatisfied"], yhat), "recall": recall_score(te["dissatisfied"], yhat),
                 "f1": f1_score(te["dissatisfied"], yhat), "threshold": float(thr)}
            mlflow.log_params({"model": name, "train_rows": len(tr), "test_rows": len(te),
                               "positive_rate_test": round(te["dissatisfied"].mean(), 4)})
            mlflow.log_metrics(m)
            info = log_sklearn(pipe, tr[NUM + CAT].head(100), pipe.predict(tr[NUM + CAT].head(100)))
            rows.append({"model": name, **m, "model_uri": info.model_uri})
            probs[name] = (p, yhat)
    res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)   # PR-AUC - для незбалансованих класів
    best = res.iloc[0]
    mv = mlflow.register_model(best.model_uri, "ecom_customer_dissatisfaction")
    promote("ecom_customer_dissatisfaction", mv.version)
    res.drop(columns="model_uri").to_csv(REPORTS / "metrics_dissatisfaction.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (p, _) in probs.items():
        fpr, tpr, _ = roc_curve(te["dissatisfied"], p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(te['dissatisfied'], p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="#999"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    cm = confusion_matrix(te["dissatisfied"], probs[best.model][1])
    ax[1].imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm):
        ax[1].text(b, a, f"{v}", ha="center", va="center", fontsize=12, color="black" if v < cm.max() / 2 else "white")
    ax[1].set_xticks([0, 1], ["задоволений", "незадоволений"]); ax[1].set_yticks([0, 1], ["задоволений", "незадоволений"])
    ax[1].set_xlabel("Прогноз"); ax[1].set_ylabel("Факт")
    fig.tight_layout(); fig.savefig(FIG / "dissatisfaction_roc_cm.png", dpi=200); plt.close(fig)
    return res, cm


if __name__ == "__main__":
    sil, prof, _ = segmentation(); print(sil); print(prof)
    res, cm = dissatisfaction(); print(res); print(cm)
