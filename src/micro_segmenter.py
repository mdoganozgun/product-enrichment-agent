# src/micro_segmenter.py
from __future__ import annotations

import os
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_customer_embeddings(path: str) -> pd.DataFrame:
    """
    Load customer embeddings pickle with columns: ['CustomerID', 'embedding'].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    df = pd.read_pickle(path)
    required = {"CustomerID", "embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Embeddings file missing columns: {sorted(missing)}")
    # enforce types
    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(str)
    return df


def to_feature_matrix(df: pd.DataFrame, col: str = "embedding") -> np.ndarray:
    """
    Stack embedding column into a 2D numpy array.
    """
    X = np.vstack(df[col].values)
    return X


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    n_init: str | int = "auto",
) -> Tuple[np.ndarray, float]:
    """
    Fit KMeans and return (labels, silhouette_score).
    If only one cluster detected or too few samples, silhouette = NaN.
    """
    if X.shape[0] < n_clusters:
        raise ValueError("Number of samples is smaller than n_clusters.")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
    return labels, sil


def try_find_category_column(df_orders: pd.DataFrame) -> Optional[str]:
    """
    Try to locate a suitable categorical column to summarize clusters.
    Preference order: 'category', 'sub_category', 'Description'.
    """
    for c in ["category", "sub_category", "Category", "SubCategory", "Description"]:
        if c in df_orders.columns:
            return c
    return None


def label_clusters_by_orders(
    clusters_df: pd.DataFrame,
    orders_csv_path: str,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Join cluster assignments with orders and derive simple textual labels:
    top-k purchased categories (or descriptions) per cluster.

    clusters_df: columns ['CustomerID', 'MicroCluster_KMeans_k{n}', 'embedding']
    returns clusters_df + ['ClusterLabel'] (string) + ['TopItemsJSON'] (json list)
    """
    if not os.path.exists(orders_csv_path):
        print(f"[info] Orders file not found: {orders_csv_path}. Skipping labeling.")
        clusters_df["ClusterLabel"] = ""
        clusters_df["TopItemsJSON"] = "[]"
        return clusters_df

    orders = pd.read_csv(orders_csv_path)
    if "CustomerID" not in orders.columns:
        print("[info] 'CustomerID' missing in orders file. Skipping labeling.")
        clusters_df["ClusterLabel"] = ""
        clusters_df["TopItemsJSON"] = "[]"
        return clusters_df

    orders = orders.copy()
    orders["CustomerID"] = orders["CustomerID"].astype(str)

    # find a category-like column
    cat_col = try_find_category_column(orders)
    if cat_col is None:
        print("[info] No category/description column found. Skipping labeling.")
        clusters_df["ClusterLabel"] = ""
        clusters_df["TopItemsJSON"] = "[]"
        return clusters_df

    # merge clusters with orders
    # (keep only CustomerID and chosen cluster column to avoid column name ambiguity)
    cluster_col = [c for c in clusters_df.columns if c.startswith("MicroCluster_KMeans_k")]
    if not cluster_col:
        print("[info] No cluster column found in clusters_df. Skipping labeling.")
        clusters_df["ClusterLabel"] = ""
        clusters_df["TopItemsJSON"] = "[]"
        return clusters_df
    cluster_col = cluster_col[0]

    merged = orders[["CustomerID", cat_col]].merge(
        clusters_df[["CustomerID", cluster_col]], on="CustomerID", how="inner"
    )

    labels = {}
    top_items_json = {}

    for clu, g in merged.groupby(cluster_col):
        # count by category (or description)
        counts = g[cat_col].value_counts().head(top_k)
        items = counts.index.tolist()
        # simple label: join top items (truncate for brevity)
        label = ", ".join(items[:3]) if items else ""
        labels[clu] = label
        top_items_json[clu] = json.dumps(items, ensure_ascii=False)

    # map back to clusters_df
    clusters_df["ClusterLabel"] = clusters_df[cluster_col].map(labels).fillna("")
    clusters_df["TopItemsJSON"] = clusters_df[cluster_col].map(top_items_json).fillna("[]")
    return clusters_df


def save_micro_segments(
    df_clusters: pd.DataFrame,
    path_basic: str,
    path_labeled: Optional[str] = None,
) -> None:
    """
    Save micro segment results.
    """
    _ensure_dir(path_basic)
    df_clusters.to_csv(path_basic, index=False)
    print(f"✅ Micro-segments saved to {path_basic}")
    if path_labeled:
        _ensure_dir(path_labeled)
        df_clusters.to_csv(path_labeled, index=False)
        print(f"✅ Labeled micro-segments saved to {path_labeled}")


def main(
    embeddings_pkl: str = "../data/customer_embeddings.pkl",
    output_csv: str = "../data/customer_micro_segments.csv",
    output_labeled_csv: Optional[str] = "../data/customer_micro_segments_labeled.csv",
    orders_csv_for_labels: Optional[str] = "../data/enriched_retail.csv",
    n_clusters: int = 5,
    l2_normalize: bool = True,
) -> None:
    """
    End-to-end micro segmentation:
      1) load embeddings
      2) (optional) L2 normalize vectors
      3) KMeans -> cluster ids + silhouette
      4) save CSV
      5) (optional) derive text labels from orders data
    """
    print(f"[info] Loading embeddings from: {embeddings_pkl}")
    df = load_customer_embeddings(embeddings_pkl)

    X = to_feature_matrix(df, "embedding")
    if l2_normalize:
        X = normalize(X, norm="l2")
        print("[info] Applied L2 normalization to embeddings.")

    print(f"[info] Running KMeans (k={n_clusters}) on {X.shape[0]} customers...")
    labels, sil = run_kmeans(X, n_clusters=n_clusters)
    print(f"[info] Silhouette score (KMeans, k={n_clusters}): {sil:.4f}" if not np.isnan(sil) else
          "[info] Silhouette score not available (single cluster).")

    # attach labels
    cluster_col = f"MicroCluster_KMeans_k{n_clusters}"
    df_out = df.copy()
    df_out[cluster_col] = labels

    # optional labeling via orders file
    if orders_csv_for_labels:
        print(f"[info] Deriving cluster labels from orders: {orders_csv_for_labels}")
        df_out = label_clusters_by_orders(
            clusters_df=df_out,
            orders_csv_path=orders_csv_for_labels,
            top_k=5,
        )

    # save
    save_micro_segments(
        df_clusters=df_out.drop(columns=["embedding"]),
        path_basic=output_csv,
        path_labeled=output_labeled_csv,
    )


if __name__ == "__main__":
    # Example: adjust arguments as needed
    main(
        embeddings_pkl="../data/customer_embeddings.pkl",
        output_csv="../data/customer_micro_segments.csv",
        output_labeled_csv="../data/customer_micro_segments_labeled.csv",
        orders_csv_for_labels="../data/enriched_retail.csv",
        n_clusters=4,
        l2_normalize=True,
    )
