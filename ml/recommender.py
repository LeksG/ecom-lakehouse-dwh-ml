"""ML-задача 3: рекомендаційна система (item-based колаборативна фільтрація на рівні категорій).

Матриця «покупець x категорія» (неявний зворотний зв'язок - факт покупки) -> матриця спільних покупок,
схожість sim(i, j) = P(j | i) = co(i, j) / n(i) (умовна ймовірність купівлі j серед покупців i). Для покупця рекомендуються top-N категорій, найбільш схожих на вже придбані,
з доповненням популярністю у штаті покупця (гібрид для «холодного старту»).
Оцінка: leave-last-out на покупцях із >= 2 замовленнями, метрики HitRate@K та MRR@K
порівняно з baseline «глобальна популярність».
"""
import json

import mlflow
import numpy as np
import pandas as pd
from scipy import sparse
from sqlalchemy import create_engine

from config import PG_DSN, REPORTS
from ml.mlflow_utils import setup

K = 5


def load_history(eng):
    q = """select c.customer_unique_id, c.state, p.category_en, d.full_date, f.order_id
           from dw.fact_sales f
           join dw.dim_customer c on c.customer_key=f.customer_key
           join dw.dim_product p on p.product_key=f.product_key
           join dw.dim_date d on d.date_key=f.date_key
           join dw.dim_order_status s on s.status_key=f.status_key
           where s.is_revenue"""
    return pd.read_sql(q, eng, parse_dates=["full_date"])


class CategoryRecommender:
    def fit(self, h: pd.DataFrame):
        self.cats = sorted(h["category_en"].unique())
        self.ci = {c: i for i, c in enumerate(self.cats)}
        users = h["customer_unique_id"].astype("category")
        m = sparse.csr_matrix((np.ones(len(h)), (users.cat.codes, h["category_en"].map(self.ci))),
                              shape=(users.cat.categories.size, len(self.cats)))
        m.data = np.minimum(m.data, 1)
        co = (m.T @ m).toarray().astype(float)          # кількість покупців, що купували i та j
        n = np.diag(co).copy()
        self.sim = co / np.maximum(n[:, None], 1)       # P(j | i)
        np.fill_diagonal(self.sim, 0)
        self.pop = h.groupby("category_en")["order_id"].nunique().reindex(self.cats).fillna(0).values
        self.pop = self.pop / self.pop.max()
        self.state_pop = (h.groupby(["state", "category_en"])["order_id"].nunique()
                           .unstack(fill_value=0).reindex(columns=self.cats, fill_value=0))
        self.state_pop = self.state_pop.div(self.state_pop.max(axis=1), axis=0)
        return self

    def recommend(self, bought: list[str], state: str | None = None, k: int = K, alpha: float = 0.5):
        idx = [self.ci[c] for c in bought if c in self.ci]
        score = self.sim[idx].sum(axis=0) if idx else np.zeros(len(self.cats))
        if score.max() > 0:
            score = score / score.max()
        pop = self.state_pop.loc[state].values if state in self.state_pop.index else self.pop
        score = alpha * score + (1 - alpha) * pop
        score[idx] = -1          # не рекомендувати вже придбане
        return [self.cats[i] for i in np.argsort(-score)[:k]]


def evaluate(h, alpha=0.5):
    h = h.sort_values("full_date")
    orders = h.groupby(["customer_unique_id", "order_id"]).agg(date=("full_date", "min"),
                                                                cats=("category_en", lambda x: sorted(set(x))),
                                                                state=("state", "first")).reset_index()
    multi = orders.groupby("customer_unique_id").filter(lambda x: len(x) >= 2)
    last = multi.sort_values("date").groupby("customer_unique_id").tail(1)
    test_orders = set(last["order_id"])
    train_h = h[~h["order_id"].isin(test_orders)]
    rec = CategoryRecommender().fit(train_h)
    hist = train_h.groupby("customer_unique_id")["category_en"].apply(lambda x: sorted(set(x)))
    pop_rank = list(pd.Series(rec.pop, index=rec.cats).sort_values(ascending=False).index)
    res = {"model": [], "base": []}
    rows = 0
    for _, r in last.iterrows():
        bought = hist.get(r.customer_unique_id, [])
        target = set(r.cats) - set(bought)
        if not target or not bought:
            continue          # оцінюємо лише відкриття нових категорій
        rows += 1
        for key, recs in [("model", rec.recommend(bought, r.state, alpha=alpha)),
                          ("base", [c for c in pop_rank if c not in bought][:K])]:
            ranks = [i + 1 for i, c in enumerate(recs) if c in target]
            res[key].append((1 if ranks else 0, 1 / ranks[0] if ranks else 0))
    out = {}
    for key, v in res.items():
        a = np.array(v)
        out[f"{key}_hit_rate_at_{K}"] = float(a[:, 0].mean())
        out[f"{key}_mrr_at_{K}"] = float(a[:, 1].mean())
    out["eval_customers"] = rows
    return out


def train():
    setup("recommender")
    eng = create_engine(PG_DSN)
    h = load_history(eng)
    grid = {}
    for a in [1.0, 0.7, 0.5, 0.3]:
        with mlflow.start_run(run_name=f"item_condprob_alpha{a}"):
            grid[a] = evaluate(h, a)
            mlflow.log_params({"algorithm": "item-based P(j|i) + state popularity", "k": K, "alpha": a})
            mlflow.log_metrics(grid[a])
    best_alpha = 0.5
    m = {**grid[best_alpha], "alpha": best_alpha}
    with mlflow.start_run(run_name="item_condprob_final"):
        mlflow.log_params({"algorithm": "item-based P(j|i) + state popularity", "k": K, "alpha": best_alpha})
        mlflow.log_metrics(m)
        rec = CategoryRecommender().fit(h)
        top_pairs = []
        for i, c in enumerate(rec.cats):
            j = int(np.argmax(rec.sim[i]))
            top_pairs.append({"category": c, "most_similar": rec.cats[j], "p_j_given_i": round(float(rec.sim[i, j]), 3)})
        pairs = pd.DataFrame(top_pairs).sort_values("p_j_given_i", ascending=False)
        pairs.to_csv(REPORTS / "recommender_pairs.csv", index=False)
        mlflow.log_artifact(str(REPORTS / "recommender_pairs.csv"))
        (REPORTS / "metrics_recommender.json").write_text(json.dumps({"final": m, "grid": grid}, indent=2))
    return m, rec, h


if __name__ == "__main__":
    m, _, _ = train(); print(m)
