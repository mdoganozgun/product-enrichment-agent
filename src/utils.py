# util.py
import os
import math
from typing import Iterable, Tuple, Optional, Sequence, List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# =========================
# I/O & Validation
# =========================
def load_transactions(csv_path: str) -> pd.DataFrame:
    """
    Load transactions CSV into a DataFrame. Raises if file is missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    return pd.read_csv(csv_path)

def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """
    Ensure required columns exist; raise ValueError if any are missing.
    """
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

# =========================
# Cleaning helpers
# =========================
def drop_na(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Drop rows with NA in the specified columns."""
    return df.dropna(subset=list(cols)).copy()

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows."""
    return df.drop_duplicates().copy()

def remove_anomalous_stockcodes(df: pd.DataFrame, stock_col: str = "StockCode") -> pd.DataFrame:
    """
    Remove rows whose StockCode contains <= 1 digit (often non-product/service-like rows).
    """
    unique_codes = df[stock_col].astype(str).unique()
    bad = {c for c in unique_codes if sum(ch.isdigit() for ch in str(c)) in (0, 1)}
    if not bad:
        return df
    return df[~df[stock_col].astype(str).isin(bad)].copy()

def remove_service_lines(df: pd.DataFrame,
                         desc_col: str = "Description",
                         services: Optional[set] = None) -> pd.DataFrame:
    """
    Remove rows known to be service lines by exact description match.
    """
    if services is None:
        services = {"Next Day Carriage", "High Resolution Image"}
    return df[~df[desc_col].isin(services)].copy()

def uppercase_descriptions(df: pd.DataFrame, desc_col: str = "Description") -> pd.DataFrame:
    """Uppercase the Description column."""
    df = df.copy()
    df[desc_col] = df[desc_col].astype(str).str.upper()
    return df

def filter_positive_prices(df: pd.DataFrame, price_col: str = "UnitPrice") -> pd.DataFrame:
    """Keep only rows with positive UnitPrice."""
    return df[df[price_col] > 0].copy()

def parse_invoice_date(df: pd.DataFrame, date_col: str = "InvoiceDate") -> pd.DataFrame:
    """Parse InvoiceDate to datetime and drop rows with invalid dates."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=[date_col])

def add_invoice_day(df: pd.DataFrame, date_col: str = "InvoiceDate") -> pd.DataFrame:
    """Add date-only helper column 'InvoiceDay' from InvoiceDate."""
    df = df.copy()
    df["InvoiceDay"] = df[date_col].dt.date
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    A lightweight cleaning pipeline composed of atomic helpers above.
    """
    required_cols = {"CustomerID", "Description", "StockCode", "InvoiceNo", "Quantity", "UnitPrice", "InvoiceDate"}
    ensure_required_columns(df, required_cols)

    df = drop_na(df, ["CustomerID", "Description"])
    df = drop_duplicates(df)
    df = remove_anomalous_stockcodes(df, "StockCode")
    df = remove_service_lines(df, "Description")
    df = uppercase_descriptions(df, "Description")
    df = filter_positive_prices(df, "UnitPrice")
    df = parse_invoice_date(df, "InvoiceDate")
    df = add_invoice_day(df, "InvoiceDate")
    df.reset_index(drop=True, inplace=True)
    return df

# =========================
# Feature matrix / Modeling helpers
# =========================
def build_feature_matrix(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Strict NA drop; select all numeric columns except CustomerID; return (df_numeric, X).
    """
    df_num = df_features.copy()
    df_num = df_num.dropna(axis=0, how="any").copy()

    numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]
    if "CustomerID" in numeric_cols:
        numeric_cols.remove("CustomerID")
    if not numeric_cols:
        raise ValueError("No numeric columns available to build features.")

    X = df_num[numeric_cols].astype(float).values
    return df_num, X

def run_isolation_forest(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Fit IsolationForest; return boolean mask of inliers (True=inlier).
    """
    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X)  # 1=inlier, -1=outlier
    return labels == 1

def kmeans_cluster(X: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, float]:
    """
    Run KMeans and return (labels, silhouette_score).
    """
    if X.shape[0] < k:
        raise ValueError("Number of samples is smaller than k.")
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
    return labels, score

def scan_kmeans_with_isoforest_plot(
    X: np.ndarray,
    k_values: List[int],
    random_state: int = 42,
    show_plot: bool = True,
    save_path: str | None = None,
) -> Dict[str, Any]:
    inlier_mask = run_isolation_forest(X, random_state=random_state)
    X_in = X[inlier_mask]
    n_in = X_in.shape[0]

    k_grid = [int(k) for k in k_values if isinstance(k, (int, np.integer)) and 2 <= int(k) <= n_in]
    if not k_grid:
        raise ValueError("Geçerli k bulunamadı: k en az 2 ve inlier sayısından küçük/eşit olmalı.")

    inertias: List[float] = []
    silhouettes: List[float] = []

    # k başına tek fit
    labels_per_k: dict[int, np.ndarray] = {}
    for k in k_grid:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(X_in)
        labels_k = km.labels_
        labels_per_k[k] = labels_k
        inertias.append(float(km.inertia_))
        sil_k = silhouette_score(X_in, labels_k) if len(set(labels_k)) > 1 else float("-inf")
        silhouettes.append(float(sil_k))

    best_idx = int(np.argmax(silhouettes))
    best_k = k_grid[best_idx]
    final_labels_in = labels_per_k[best_k]
    best_sil = float(silhouettes[best_idx])

    labels_full = np.full(X.shape[0], -1, dtype=int)
    labels_full[np.where(inlier_mask)[0]] = final_labels_in

    title = f"KMeans on Inliers (best k={best_k}, silhouette={best_sil:.3f})"
    plot_2d(X, labels_full, title=title, show=show_plot, save_path=save_path)

    return {
        "best_k": best_k,
        "best_silhouette": best_sil,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "k_grid": k_grid,
        "labels_full": labels_full,
        "inlier_mask": inlier_mask,
    }

# =========================
# Batching / API helpers
# =========================
def chunk_iter(seq: List[str], size: int) -> Iterable[List[str]]:
    """Yield consecutive chunks of size 'size' from seq."""
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def total_batches(n_items: int, batch_size: int) -> int:
    """Compute number of batches for n_items and batch_size."""
    return max(1, math.ceil(n_items / max(1, batch_size)))

def resolve_openai_api_key(explicit_key: Optional[str]) -> str:
    """
    Return the provided key if set; otherwise read OPENAI_API_KEY from environment.
    Raise if neither exists.
    """
    key = explicit_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key is not set. Provide api_key or set OPENAI_API_KEY.")
    return key

# =========================
# Text helpers for product embedding
# =========================
DEFAULT_TEXT_COLS = ["Description", "category", "sub_category", "usage_context"]

def build_product_text(row: pd.Series,
                       text_cols: Optional[Iterable[str]] = None,
                       tags_col: str = "tags") -> str:
    """
    Build a single text string from a product row to feed into a text-embedding model.
    - Concatenates non-empty columns listed in text_cols (defaults to DEFAULT_TEXT_COLS)
    - If tags column is a list, append non-empty tags as space-separated tokens
    """
    cols = list(text_cols) if text_cols is not None else DEFAULT_TEXT_COLS
    parts: List[str] = []

    for col in cols:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]).strip())

    if tags_col in row:
        tags = row[tags_col]
        if isinstance(tags, list) and tags:
            parts.extend([str(t).strip() for t in tags if t])

    return " ".join([p for p in parts if p])

def build_product_texts(df: pd.DataFrame, **kwargs) -> List[str]:
    """Apply build_product_text to each row and return the list of strings."""
    return [build_product_text(r, **kwargs) for _, r in df.iterrows()]

# =========================
# Numeric / similarity helpers
# =========================
def half_life_decay(invoice_dates: pd.Series, half_life_days: int) -> np.ndarray:
    """
    Compute per-row exponential time decay weights using a half-life in days.
    """
    max_date = pd.to_datetime(invoice_dates).max()
    days = (max_date - pd.to_datetime(invoice_dates)).dt.days.clip(lower=0)
    return np.power(0.5, days / float(half_life_days))

def weighted_average_stack(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted average over a stack of vectors with row-aligned weights.
    If all weights are zero, falls back to simple mean.
    """
    if vectors.ndim == 1:  # (d,)
        vectors = vectors[None, :]
    if np.all(weights == 0):
        return vectors.mean(axis=0)
    return np.average(vectors, axis=0, weights=weights)

def unit_normalize(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize a matrix; safe against zero vectors."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return M / norms

def cosine_scores(query: np.ndarray, catalog: np.ndarray) -> np.ndarray:
    """
    Fast cosine similarity via normalized dot-product between query (d,) and catalog (n,d).
    """
    q = unit_normalize(query.reshape(1, -1))
    C = unit_normalize(catalog)
    return (q @ C.T).ravel()

def topk_indices_desc(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k values in 'scores' in descending order.
    """
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k-1)[:k]
    return idx[np.argsort(-scores[idx])]

# =========================
# Category profiling helpers
# =========================
def top_categories_for_customer(
    df: pd.DataFrame,
    customer_id: str | int,
    n: int = 5,
    category_col: str = "category",
    qty_col: str = "Quantity",
    use_quantity: bool = True,
    half_life_days: int | None = None,
) -> pd.DataFrame:
    """
    Müşterinin en güçlü kategorilerini döndürür.
    Skor = (Quantity>=0) * (opsiyonel yarı-ömür çürümesi) toplamı.
    Kolonlar: [category_col, score, count, rank]
    """
    # Koruma: gerekli kolonlar var mı?
    if "CustomerID" not in df.columns or category_col not in df.columns:
        return pd.DataFrame(columns=[category_col, "score", "count", "rank"])

    h = df[df["CustomerID"].astype(str) == str(customer_id)]
    if h.empty:
        return pd.DataFrame(columns=[category_col, "score", "count", "rank"])

    # Ağırlık: miktar veya 1.0
    if use_quantity and qty_col in h.columns:
        w = np.clip(h[qty_col].astype(float), 0.0, None)
    else:
        w = pd.Series(1.0, index=h.index)

    # Zaman çürümesi
    if half_life_days is not None and "InvoiceDate" in h.columns:
        decay = half_life_decay(h["InvoiceDate"], half_life_days)  # np.ndarray
        w = w * pd.Series(decay, index=h.index)

    # Geçici ağırlık kolonu ekle ve grupla
    h2 = h.copy()
    h2["_w"] = w.values

    grouped = (
        h2.groupby(category_col, dropna=False)
          .agg(score=("_w", "sum"), count=(category_col, "size"))
          .reset_index()
          .sort_values(by=["score", "count", category_col], ascending=[False, False, True])
    )

    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped.head(n).reset_index(drop=True)


def top_subcategories_for_customer(
    df: pd.DataFrame,
    customer_id: str | int,
    n: int = 5,
    subcategory_col: str = "sub_category",
    qty_col: str = "Quantity",
    use_quantity: bool = True,
    half_life_days: int | None = None,
) -> pd.DataFrame:
    """
    Müşterinin en güçlü alt-kategorilerini döndürür.
    Skor = (Quantity>=0) * (opsiyonel yarı-ömür çürümesi) toplamı.
    Kolonlar: [subcategory_col, score, count, rank]
    """
    if "CustomerID" not in df.columns or subcategory_col not in df.columns:
        return pd.DataFrame(columns=[subcategory_col, "score", "count", "rank"])

    h = df[df["CustomerID"].astype(str) == str(customer_id)]
    if h.empty:
        return pd.DataFrame(columns=[subcategory_col, "score", "count", "rank"])

    if use_quantity and qty_col in h.columns:
        w = np.clip(h[qty_col].astype(float), 0.0, None)
    else:
        w = pd.Series(1.0, index=h.index)

    if half_life_days is not None and "InvoiceDate" in h.columns:
        decay = half_life_decay(h["InvoiceDate"], half_life_days)
        w = w * pd.Series(decay, index=h.index)

    h2 = h.copy()
    h2["_w"] = w.values

    grouped = (
        h2.groupby(subcategory_col, dropna=False)
          .agg(score=("_w", "sum"), count=(subcategory_col, "size"))
          .reset_index()
          .sort_values(by=["score", "count", subcategory_col], ascending=[False, False, True])
    )

    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped.head(n).reset_index(drop=True)

# =========================
# PCA & plotting helpers
# =========================
def pca_project(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Safe PCA projection to 2D for visualization. Pads tiny inputs if needed.
    """
    if X.ndim != 2:
        X = np.asarray(X).reshape(len(X), -1)
    n_samples, n_features = X.shape
    n = min(n_samples, n_features)
    if n < n_components:
        # pad to avoid PCA errors on very small inputs
        pad_samples = max(0, n_components - n_samples)
        pad_features = max(0, n_components - n_features)
        if pad_samples or pad_features:
            X = np.pad(X, ((0, pad_samples), (0, pad_features)), mode="edge")
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)

def plot_2d(X: np.ndarray, labels: np.ndarray, title: str | None = None,
            show: bool = True, save_path: str | None = None) -> None:
    proj = pca_project(X, 2)
    plt.figure(figsize=(8, 6))
    for lab in np.unique(labels):
        mask = labels == lab
        name = f"Cluster {lab}" if lab != -1 else "Outlier"
        plt.scatter(proj[mask, 0], proj[mask, 1], s=30, alpha=0.85, label=name)
    if title:
        plt.title(title)
    plt.xlabel("PC 1"); plt.ylabel("PC 2"); plt.grid(True, alpha=0.3); plt.legend()

    if save_path:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def get_customer_orders_summary(customer_id: int, df: pd.DataFrame,
                                product_col="Description", category_col="Category"):
    """
    Belirli bir müşteri için siparişleri, toplam harcama özetini
    ve en çok harcama yapılan ürün/kategorileri döndürür.
    """
    # Müşteri siparişlerini filtrele
    customer_orders = df[df["CustomerID"] == customer_id].copy()
    if customer_orders.empty:
        print(f"CustomerID {customer_id} için sipariş bulunamadı.")
        return None

    # Toplam harcama
    customer_orders["LineTotal"] = customer_orders["Quantity"] * customer_orders["UnitPrice"]
    total_spent = customer_orders["LineTotal"].sum()

    # Sipariş bazlı toplamlar
    order_totals = customer_orders.groupby("InvoiceNo")["LineTotal"].sum()

    # En çok harcama yapılan ürünler
    top_products = (
        customer_orders.groupby(product_col)["LineTotal"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    # En çok harcama yapılan kategoriler
    if category_col in df.columns:
        top_categories = (
            customer_orders.groupby(category_col)["LineTotal"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
    else:
        top_categories = None

    print(f"CustomerID {customer_id} için toplam harcama: {total_spent:.2f}")
    print("\nSipariş bazlı toplamlar:")
    print(order_totals)

    print("\nEn çok harcama yapılan ürünler:")
    print(top_products)

    if top_categories is not None:
        print("\nEn çok harcama yapılan kategoriler:")
        print(top_categories)

    return customer_orders

def build_behavioral_cluster_profiles(
    reports_csv="../../data/customer_reports.csv",
    orders_csv="../../data/enriched_retail.csv",
    half_life_days=365,
    top_n=3
):
    import numpy as np
    import pandas as pd

    # veriler
    df_rep = pd.read_csv(reports_csv)
    df_ord = pd.read_csv(orders_csv)

    # cluster kolonunu bul
    beh_cols = [c for c in df_rep.columns if c.startswith("BehavioralCluster_")]
    if not beh_cols:
        raise ValueError("Raporda 'BehavioralCluster_' ile başlayan bir sütun bulunamadı.")
    cluster_col = beh_cols[0]

    # tipler
    df_rep["CustomerID"] = df_rep["CustomerID"].astype(str)
    df_ord["CustomerID"] = df_ord["CustomerID"].astype(str)

    # tarih
    df_ord["InvoiceDate"] = pd.to_datetime(df_ord["InvoiceDate"], errors="coerce")
    df_ord = df_ord.dropna(subset=["InvoiceDate"]).copy()
    max_date = df_ord["InvoiceDate"].max()
    days = (max_date - df_ord["InvoiceDate"]).dt.days.clip(lower=0)
    decay = np.power(0.5, days / float(half_life_days))

    # join
    df = df_ord.merge(df_rep[["CustomerID", cluster_col]], on="CustomerID", how="left")
    df = df.dropna(subset=[cluster_col]).copy()
    df[cluster_col] = df[cluster_col].astype(int)

    # ağırlık
    qty = np.clip(df["Quantity"].astype(float), 0.0, None)
    df["w"] = qty * decay

    # yardımcı
    def _topn(group, col, n=top_n):
        s = group.groupby(col)["w"].sum().sort_values(ascending=False)
        return s.head(n)

    # cluster bazlı profiller
    profiles = []
    for k, g in df.groupby(cluster_col):
        top_cat = _topn(g, "category", top_n)
        top_sub = _topn(g, "sub_category", top_n)

        cat_weights = g.groupby("category")["w"].sum()
        p = (cat_weights / cat_weights.sum()).values
        entropy = float(-(p * np.log(p + 1e-12)).sum()) if cat_weights.sum() > 0 else 0.0
        mean_days = float((g["w"] * (max_date - g["InvoiceDate"]).dt.days).sum() / (g["w"].sum() + 1e-12))

        top_cat_names = top_cat.index.tolist()
        if len(top_cat_names) >= 2:
            label = f"{top_cat_names[0]} & {top_cat_names[1]} Odaklı"
        elif len(top_cat_names) == 1:
            label = f"{top_cat_names[0]} Odaklı"
        else:
            label = "Genelci"

        profiles.append({
            "cluster": int(k),
            "label": label,
            "top_categories": ", ".join(top_cat.index.astype(str)),
            "top_categories_weights": ", ".join([f"{v:.1f}" for v in top_cat.values]),
            "top_sub_categories": ", ".join(top_sub.index.astype(str)),
            "top_sub_categories_weights": ", ".join([f"{v:.1f}" for v in top_sub.values]),
            "diversity_entropy": round(entropy, 4),
            "freshness_mean_days": round(mean_days, 1),
            "cluster_size_customers": int(df_rep[df_rep[cluster_col] == k].shape[0]),
        })

    return pd.DataFrame(profiles).sort_values("cluster").reset_index(drop=True)

import pandas as pd

def category_stats_list(df: pd.DataFrame):
    # Gerekli sütunlar: category, InvoiceNo, CustomerID, Quantity, UnitPrice
    needed = {"category", "InvoiceNo", "CustomerID", "Quantity", "UnitPrice"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik sütunlar: {sorted(missing)}")

    tmp = df.copy()
    tmp["Total_Spend"] = tmp["Quantity"].astype(float) * tmp["UnitPrice"].astype(float)

    agg = (
        tmp.groupby("category")
           .agg(
               unique_orders=("InvoiceNo", "nunique"),
               unique_customers=("CustomerID", "nunique"),
               total_items=("Quantity", "sum"),
               total_spend=("Total_Spend", "sum"),
           )
           .reset_index()
           .sort_values(["total_spend", "total_items"], ascending=False)
    )

    # İstersen doğrudan DataFrame döndürebilirsin:
    # return agg

    # “liste” olarak döndür (list of dicts)
    return agg.to_dict(orient="records")


# --- Normalizasyon yardımcıları ---
def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = s.replace("&", "and")
    s = " ".join(s.split())
    return s

TR_REPLACEMENTS = {
    "ö": "o", "ü": "u", "ğ": "g", "ş": "s", "ı": "i", "ç": "c",
    "â": "a", "î": "i", "û": "u",
}
def _rm_tr_chars(s: str) -> str:
    if not isinstance(s, str):
        return s
    return "".join(TR_REPLACEMENTS.get(ch, ch) for ch in s)

# --- Çakışma eşlemeleri (TR→ENG + ENG varyantları) ---
SYNONYMS_TO_CANON = {
    # --- Home & Yaşam (hepsi -> Home & Living) ---
    "home decor": "Home & Living",
    "ev dekorasyonu": "Home & Living",
    "home goods": "Home & Living",
    "ev gerecleri": "Home & Living",
    "home and living": "Home & Living",
    "home & living": "Home & Living",
    "home living": "Home & Living",
    "home organization": "Home & Living",
    "home and garden": "Home & Living",
    "home & garden": "Home & Living",
    "home improvement": "Home & Living",

    # --- Mutfak (hepsi -> Kitchen & Dining) ---
    "home and kitchen": "Kitchen & Dining",
    "home & kitchen": "Kitchen & Dining",
    "kitchen and dining": "Kitchen & Dining",
    "kitchen & dining": "Kitchen & Dining",
    "kitchenware": "Kitchen & Dining",
    "kitchen": "Kitchen & Dining",
    "mutfak and sofra": "Kitchen & Dining",
    "mutfak & sofra": "Kitchen & Dining",

    # --- Ofis & Kırtasiye (hepsi -> Stationery & Office Supplies) ---
    "stationery": "Stationery & Office Supplies",
    "educational supplies": "Stationery & Office Supplies",
    "office supplies": "Stationery & Office Supplies",
    "stationery and office": "Stationery & Office Supplies",
    "stationery & office": "Stationery & Office Supplies",
    "kirtasiye": "Stationery & Office Supplies",

    # --- Oyuncak & Hobi ---
    # Oyuncaklar -> Toys & Games
    "toys": "Toys & Games",
    "toys and games": "Toys & Games",
    "toys & games": "Toys & Games",
    "games and toys": "Toys & Games",
    "games & toys": "Toys & Games",
    "kids": "Toys & Games",

    # El işi / Hobi -> Arts & Crafts
    "arts and crafts": "Arts & Crafts",
    "arts & crafts": "Arts & Crafts",
    "craft supplies": "Arts & Crafts",
    "crafts": "Arts & Crafts",
    "hobby and craft": "Arts & Crafts",
    "hobby & craft": "Arts & Crafts",
    "hobbies": "Arts & Crafts",
    "hobi malzemeleri": "Arts & Crafts",

    # --- Moda & Aksesuar (hepsi -> Fashion & Accessories) ---
    "fashion": "Fashion & Accessories",
    "clothing": "Fashion & Accessories",
    "accessories": "Fashion & Accessories",
    "fashion accessories": "Fashion & Accessories",
    "personal accessories": "Fashion & Accessories",
    "jewelry": "Fashion & Accessories",

    # --- Sağlık & Kişisel Bakım (hepsi -> Health & Personal Care) ---
    "health and wellness": "Health & Personal Care",
    "health & wellness": "Health & Personal Care",
    "health and beauty": "Health & Personal Care",
    "health & beauty": "Health & Personal Care",
    "personal care": "Health & Personal Care",
    "kisisel bakim": "Health & Personal Care",
    "bath and beauty": "Health & Personal Care",
    "bath & beauty": "Health & Personal Care",

    # --- Diğerleri (ayrı kalır) ---
    "party supplies": "Party Supplies",
    "parti malzemeleri": "Party Supplies",
    "gift supplies": "Gift Supplies",
    "food and beverage": "Food & Beverage",
    "food & beverage": "Food & Beverage",
    "sports and outdoors": "Sports & Outdoors",
    "sports & outdoors": "Sports & Outdoors",
    "travel": "Travel",
    "travel accessories": "Travel",
    "electronics": "Electronics",
    "gardening": "Gardening",
    "musical instruments": "Musical Instruments",
    "pet supplies": "Pet Supplies",
}

#
# --- Standartlaştırıcı yardımcı (tek noktadan) ---

def _standardize_series(values: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    """
    TR karakter sadeleştirme + lowercase + `&` -> `and` normalizasyonu uygular,
    ardından `mapping` sözlüğüne göre kanonik etikete çevirir.
    Bilinmeyenler için normalize edilmiş ham değeri döndürür; boşlar NaN yapılır.
    """
    normed = values.astype(str).map(_rm_tr_chars).map(_norm)
    mapped = normed.map(lambda x: mapping.get(x, x))
    return mapped.replace({"": np.nan})

# --- Kategori standardizasyonu (DataFrame in-place) ---

def standardize_categories_inplace(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    """`df[category_col]` kolonunu `SYNONYMS_TO_CANON` ile normalize eder (in-place)."""
    if category_col in df.columns:
        df[category_col] = _standardize_series(df[category_col], SYNONYMS_TO_CANON)
    return df

# --- Alt-kategori eşleme sözlüğü ---
SYNONYMS_SUB_TO_CANON = {
    # Candles & Holders
    "candle holder": "candles and holders",
    "candleholder": "candles and holders",
    "candlestick": "candles and holders",
    "tealight holder": "candles and holders",
    "t light holder": "candles and holders",
    "tealight": "candles and holders",
    "tealight candle": "candles and holders",
    "candles": "candles and holders",

    # Lighting
    "lantern": "lighting",
    "string lights": "lighting",
    "light garland": "lighting",
    "night light": "lighting",
    "nightlight": "lighting",
    "lamp": "lighting",
    "lights": "lighting",

    # Decorative Ornaments
    "ornament": "decorative ornaments",
    "decorative object": "decorative ornaments",
    "figurine": "decorative ornaments",
    "bauble": "decorative ornaments",
    "decoration": "decorative ornaments",
    "christmas decoration": "decorative ornaments",
    "christmas tree": "decorative ornaments",
    "decorative tree": "decorative ornaments",
    "wreath": "decorative ornaments",
    "tree topper": "decorative ornaments",
    "advent calendar": "decorative ornaments",

    # Photo Frames
    "picture frame": "photo frames",
    "photo frame": "photo frames",
    "frame": "photo frames",
    "photo display": "photo frames",
    "fotograf cercevesi": "photo frames",

    # Wall decor & hooks
    "wall art": "wall decor and hooks",
    "wall decor": "wall decor and hooks",
    "wall clock": "wall decor and hooks",
    "clock": "wall decor and hooks",
    "wall hook": "wall decor and hooks",
    "coat rack": "wall decor and hooks",
    "coat hanger": "wall decor and hooks",
    "hook": "wall decor and hooks",
    "hanger": "wall decor and hooks",

    # Textile decor
    "cushion": "textile decor",
    "cushion cover": "textile decor",
    "quilt": "textile decor",

    # Baking & Cooking
    "kitchen set": "baking and cooking",
    "kitchenware": "baking and cooking",
    "cooking set": "baking and cooking",
    "baking supplies": "baking and cooking",
    "baking cups": "baking and cooking",
    "cake tin": "baking and cooking",
    "cake case": "baking and cooking",
    "cupcake liner": "baking and cooking",
    "cake stand": "baking and cooking",
    "cookie cutter": "baking and cooking",
    "measuring spoon": "baking and cooking",
    "measuring spoons": "baking and cooking",

    # Tableware
    "cutlery set": "tableware",
    "plate": "tableware",
    "bowl": "tableware",
    "cup": "tableware",
    "fincan": "tableware",
    "mug": "tableware",
    "mug cosy": "tableware",
    "mug warmer": "tableware",
    "jug": "tableware",
    "tableware": "tableware",

    # Napkins & Tissues
    "napkin": "napkins and tissues",
    "napkins": "napkins and tissues",
    "tissue": "napkins and tissues",
    "tissues": "napkins and tissues",
    "tissue paper": "napkins and tissues",
    "tissue box": "napkins and tissues",

    # Table accessories
    "tray": "table accessories",
    "placemat": "table accessories",
    "tablecloth": "table accessories",

    # Storage (kitchen/general)
    "bread bin": "storage and containers",
    "snack box": "storage and containers",
    "lunch box": "storage and containers",
    "lunch bag": "storage and containers",
    "storage bag": "storage and containers",
    "storage box": "storage and containers",
    "storage basket": "storage and containers",
    "storage tin": "storage and containers",
    "organiser": "storage and containers",
    "container": "storage and containers",
    "jar": "storage and containers",
    "basket": "storage and containers",
    "bucket": "storage and containers",
    "pot": "storage and containers",
    "plant holder": "storage and containers",

    # Toys
    "doll": "toys",
    "plush toy": "toys",
    "playset": "toys",
    "toy storage": "toys",

    # Puzzles & Blocks
    "puzzle": "puzzles and blocks",
    "jigsaw puzzle": "puzzles and blocks",
    "building block": "puzzles and blocks",
    "blocks": "puzzles and blocks",

    # Games
    "board game": "games",
    "card game": "games",
    "dominoes": "games",
    "skittles": "games",
    "spinning top": "games",
    "marbles": "games",
    "cup and ball game": "games",

    # Arts & Crafts Kits
    "craft kit": "arts and crafts kits",
    "paint set": "arts and crafts kits",
    "art set": "arts and crafts kits",
    "drawing board": "arts and crafts kits",
    "drawing slate": "arts and crafts kits",

    # Bags & Small accessories
    "bag": "bags",
    "tote bag": "bags",
    "shopper bag": "bags",
    "shoulder bag": "bags",
    "purse": "small accessories",
    "pouch": "small accessories",
    "bag charm": "small accessories",
    "luggage tag": "small accessories",

    # Jewelry
    "necklace": "jewelry",
    "bracelet": "jewelry",
    "ring": "jewelry",
    "bangle": "jewelry",

    # Apparel & Footwear
    "socks": "apparel and footwear",
    "slippers": "apparel and footwear",
    "slipper": "apparel and footwear",
    "hat": "apparel and footwear",

    # Fashion Accessories
    "bandana": "fashion accessories",
    "hair grip": "fashion accessories",
    "hair clip": "fashion accessories",

    # Stickers & Gift wrapping
    "sticker": "stickers",
    "sticker sheet": "stickers",
    "cikartma": "stickers",
    "ribbon": "gift wrapping",
    "ribbons": "gift wrapping",
    "gift tag": "gift wrapping",
    "gift box": "gift wrapping",
    "party bag": "gift wrapping",

    # Stationery
    "notebook": "stationery",
    "writing set": "stationery",
    "stationery set": "stationery",
    "pencil": "stationery",
    "pencils": "stationery",
    "pen": "stationery",
    "kalem": "stationery",
    "eraser": "stationery",
    "scissor": "stationery",
    "card": "stationery",
    "greeting card": "stationery",
    "invitation card": "stationery",
    "card holder": "stationery",

    # Boards & Surfaces / Tools
    "chalkboard": "boards and writing",
    "blackboard": "boards and writing",
    "calculator": "office tools",
    "globe": "office tools",

    # Bath & Body Care / Comfort
    "bath sponge": "bath and body care",
    "sponge": "bath and body care",
    "sunger": "bath and body care",
    "washcloth": "bath and body care",
    "hand warmer": "warmers and comfort",
    "hot water bottle": "warmers and comfort",

    # First Aid
    "plaster": "first aid",
    "plasters": "first aid",
    "first aid": "first aid",
    "first aid kit": "first aid",

    # Entryway
    "doormat": "door and entryway",
    "doorstop": "door and entryway",
    "doorsign": "door and entryway",
    "sign": "door and entryway",

    # Furniture accessories
    "drawer knob": "furniture accessories",
    "drawerknob": "furniture accessories",
    "knob": "furniture accessories",

    # Party supplies
    "party hat": "party supplies",
    "balloon": "party supplies",
    "garland": "party supplies",
    "hanging decor": "party supplies",
    "disco ball": "party supplies",

    # Misc
    "dog bowl": "pet supplies",
    "harmonica": "misc electronics and instruments",
    "motor": "misc electronics and instruments",
    "speaker": "misc electronics and instruments",
    "panettone": "food items",
    "candy": "food items",
    "biscuit": "food items",
    "cake": "food items",
    "incense": "home fragrance",
    "umbrella": "outdoors and gardening",
    "parasol": "outdoors and gardening",
    "spade": "outdoors and gardening",
    "trellis": "outdoors and gardening",
}
# --- Alt-kategori standardizasyonu (DataFrame in-place) ---

def standardize_subcategories_inplace(df: pd.DataFrame, subcategory_col: str = "sub_category") -> pd.DataFrame:
    """`df[subcategory_col]` kolonunu `SYNONYMS_SUB_TO_CANON` ile normalize eder (in-place)."""
    if subcategory_col in df.columns:
        df[subcategory_col] = _standardize_series(df[subcategory_col], SYNONYMS_SUB_TO_CANON)
    return df

# --- Kategori + Alt kategori birlikte in-place standardizasyon ---

def normalize_category_and_subcategory_inplace(df: pd.DataFrame,
                                               category_col: str = "category",
                                               subcategory_col: str = "sub_category") -> pd.DataFrame:
    df = standardize_categories_inplace(df, category_col)
    df = standardize_subcategories_inplace(df, subcategory_col)
    return df

# Örnek:
# df_orders = pd.read_csv("orders.csv")
# df_orders = normalize_category_and_subcategory_inplace(df_orders, category_col="category", subcategory_col="sub_category")

def robust_qty_weight(qty: pd.Series,
                      decay: np.ndarray,
                      *,
                      use_log1p: bool = True,
                      cap: Optional[float] = None) -> pd.Series:
    """
    Adetleri (qty) zaman çürümesi ile ağırlıklandırırken robustlaştır:
      - use_log1p=True ise log1p(qty) kullan
      - cap varsa (ör. 10/20), loglanan ya da normal qty'yi min(cap) ile sınırla
    """
    q = qty.astype(float).clip(lower=0)
    if use_log1p:
        q = np.log1p(q)
    if cap is not None:
        q = np.minimum(q, float(cap))
    w = q * decay
    return pd.Series(w, index=qty.index)


import pandas as pd


def analyze_cluster_subcategories(
        df: pd.DataFrame,
        cluster_col: str = "BehavioralCluster_KMeans_k3",
        subcat_col: str = "sub_category",
        spend_col: str = "Total_Spend",  # harcama kolonu
        top_n: int = 5
) -> dict:
    """
    Her cluster için en çok harcama yapılan alt kategorileri bulur.

    Parameters
    ----------
    df : pd.DataFrame
        İçinde cluster_col, subcat_col ve spend_col olan DataFrame.
    cluster_col : str
        Cluster etiketlerini tutan sütun adı.
    subcat_col : str
        Alt kategoriyi tutan sütun adı.
    spend_col : str
        Harcama bilgisini tutan sütun adı.
    top_n : int
        Her cluster için kaç alt kategori gösterileceği.

    Returns
    -------
    results : dict
        {cluster_id: DataFrame} yapısında, her cluster için top-N sub_category.
    """
    results = {}
    for cluster_id, grp in df.groupby(cluster_col):
        # alt kategori bazında toplam harcama
        summary = (
            grp.groupby(subcat_col)[spend_col]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        results[cluster_id] = summary
    return results


# Optional: limit what gets imported with `from util import *`
__all__ = [
    # IO / validation
    "load_transactions", "ensure_required_columns",
    # cleaning
    "drop_na", "drop_duplicates", "remove_anomalous_stockcodes", "remove_service_lines",
    "uppercase_descriptions", "filter_positive_prices", "parse_invoice_date", "add_invoice_day", "basic_clean",
    # feature matrix / modeling
    "build_feature_matrix", "run_isolation_forest", "kmeans_cluster", "scan_kmeans_with_isoforest_plot", "build_behavioral_cluster_profiles",
    # batching / api
    "chunk_iter", "total_batches", "resolve_openai_api_key",
    # text helpers
    "DEFAULT_TEXT_COLS", "build_product_text", "build_product_texts",
    # numeric / similarity
    "half_life_decay", "weighted_average_stack", "unit_normalize", "cosine_scores", "topk_indices_desc",
    "top_categories_for_customer", "top_subcategories_for_customer", "get_customer_orders_summary", "category_stats_list",
    "standardize_categories_inplace", "SYNONYMS_TO_CANON", "SYNONYMS_SUB_TO_CANON", "pca_project", "plot_2d",
    "standardize_subcategories_inplace", "normalize_category_and_subcategory_inplace","analyze_cluster_subcategories"
]