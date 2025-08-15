# src/utils/recommendation_utils.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def _time_decay(series_dates: pd.Series, half_life_days: int) -> np.ndarray:
    dates = pd.to_datetime(series_dates, errors="coerce")
    max_date = dates.max()
    days_since = (max_date - dates).dt.days.clip(lower=0)
    return np.power(0.5, days_since / float(half_life_days))

def recommend_products_for_customer(
    customer_id,
    embeddings_path: str,
    orders_path: str,
    top_k_customers: int = 50,
    top_n_products: int = 10,
    half_life_days: int = 180,
    exclude_already_bought: bool = True,
    min_qty: int | None = 1,
):
    df_emb = pd.read_pickle(embeddings_path)
    df_ord = pd.read_csv(orders_path)

    if "CustomerID" not in df_emb.columns or "embedding" not in df_emb.columns:
        raise ValueError("Embeddings file must contain 'CustomerID' and 'embedding' columns.")

    req_cols = {"CustomerID","StockCode","Description","Quantity","UnitPrice","InvoiceDate"}
    missing = req_cols - set(df_ord.columns)
    if missing:
        raise ValueError(f"Orders file missing columns: {sorted(missing)}")

    df_emb["CustomerID"] = df_emb["CustomerID"].astype(str)
    df_ord["CustomerID"] = df_ord["CustomerID"].astype(str)
    df_ord["StockCode"]  = df_ord["StockCode"].astype(str)
    df_ord["InvoiceDate"] = pd.to_datetime(df_ord["InvoiceDate"], errors="coerce")
    df_ord = df_ord.dropna(subset=["InvoiceDate"])

    if str(customer_id) not in df_emb["CustomerID"].values:
        raise ValueError(f"Customer {customer_id} not found in embeddings.")

    target_vec = df_emb.loc[df_emb["CustomerID"] == str(customer_id), "embedding"].values[0].reshape(1, -1)
    all_vecs   = np.vstack(df_emb["embedding"].values)
    sims       = cosine_similarity(target_vec, all_vecs)[0]

    temp = df_emb.copy()
    temp["similarity"] = sims
    top_sim = (temp[temp["CustomerID"] != str(customer_id)]
               .sort_values("similarity", ascending=False)
               .head(top_k_customers))
    neighbor_ids = set(top_sim["CustomerID"])

    cand = df_ord[df_ord["CustomerID"].isin(neighbor_ids)].copy()
    if min_qty is not None:
        cand = cand[cand["Quantity"] >= min_qty]
    if cand.empty:
        return pd.DataFrame(columns=["StockCode","Description","score","TotalQty","LastPurchasedDate","AvgNeighborSimilarity"])

    decay = _time_decay(cand["InvoiceDate"], half_life_days=half_life_days)
    cand["score"] = cand["Quantity"].astype(float) * cand["UnitPrice"].astype(float) * decay

    if exclude_already_bought:
        bought = set(df_ord.loc[df_ord["CustomerID"] == str(customer_id), "StockCode"])
        cand = cand[~cand["StockCode"].isin(bought)]

    if cand.empty:
        return pd.DataFrame(columns=["StockCode","Description","score","TotalQty","LastPurchasedDate","AvgNeighborSimilarity"])

    out = (cand.groupby(["StockCode","Description"])
                .agg(score=("score","sum"),
                     TotalQty=("Quantity","sum"),
                     LastPurchasedDate=("InvoiceDate","max"))
                .reset_index()
                .sort_values("score", ascending=False)
                .head(top_n_products))

    out["AvgNeighborSimilarity"] = top_sim["similarity"].mean()
    return out