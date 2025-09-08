import os
import pandas as pd
import numpy as np
from typing import Dict
from utils import half_life_decay, weighted_average_stack, cosine_scores, topk_indices_desc, unit_normalize


class CustomerEmbedder:
    """
    Customer-level embedding builder (simplified)
    -------------------------------------------
    This class computes **one** type of customer embedding using a weighted
    average of product embeddings with **time-decay** applied.

    • Weighting: effective_quantity = Quantity * decay
    • Decay: half-life model => decay = 0.5 ** (days_since / half_life_days)
    • Final weight: effective_quantity * UnitPrice (can be disabled)

    Rationale:
    - A "1-year half-life" means: if 2 items were bought 365 days ago, they
      contribute like ~1 item today (2 * 0.5 = 1). More recent purchases
      weigh more, older purchases fade out.

    Usage:
        embedder = CustomerEmbedder(
            enriched_data_path="../data/enriched_retail.csv",
            product_embedding_path="../data/products_with_embeddings.pkl",
            output_path="../data/customer_embeddings.pkl",
        )
        embedder.generate_customer_embeddings(half_life_days=365, use_unit_price=True)
    """

    def __init__(
        self,
        enriched_data_path: str = "../../data/enriched_retail.csv",
        product_embedding_path: str = "../../data/products_with_embeddings.pkl",
        output_path: str = "../../data/customer_embeddings.pkl",
        behavioral_output_path: str = "../../data/customer_behavioral_embeddings.pkl"
    ):
        self.enriched_data_path = enriched_data_path
        self.product_embedding_path = product_embedding_path
        self.output_path = output_path
        self.behavioral_output_path = behavioral_output_path

        # Internal caches to avoid repeated IO
        self._df_products = None
        self._df_cust_emb = None
        self._df_orders = None

    def _load_orders(self) -> pd.DataFrame:
        """
        Load and cache orders from CSV. Ensures types and parses InvoiceDate.
        """
        if self._df_orders is None:
            df = pd.read_csv(self.enriched_data_path)
            df["CustomerID"] = df["CustomerID"].astype(str)
            df["StockCode"] = df["StockCode"].astype(str)
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            self._df_orders = df.dropna(subset=["InvoiceDate"]).copy()
        return self._df_orders

    # -----------------------------
    # Core API
    # -----------------------------

    def generate_behavioral_embeddings(
            self,
            half_life_days: int = 365,
            normalize: bool = True,
            output_path: str | None = "../../data/customer_behavioral_embeddings.pkl",
            *,
            # --- yeni: robust weighting parametreleri ---
            use_log1p: bool = True,
            qty_cap: float | None = None,
            # --- yeni: dinamik kategori karışımı ---
            cat_mix_alpha: float = 0.15,
            category_col: str = "category",
    ) -> pd.DataFrame:
        """
        Para/metrikleri kullanmadan, sadece satın alma *sıklığı* + *zaman çürümesi* ile
        müşteri davranış/mikro-segment embedding'leri üretir.

        Ekler:
          - Robust weighting: log1p(qty) ve/veya cap
          - Dinamik kategori centroid karışımı: müşteri dağılımına göre kategori centroid’lerini
            embedding’e (alpha oranında) karıştırır.
        """
        # 1) Veri yükle
        df_orders = self._load_orders().copy()
        df_products = pd.read_pickle(self.product_embedding_path).copy()

        # 2) Şema kontrolü
        required = {"StockCode", "CustomerID", "Quantity", "InvoiceDate"}
        missing = required - set(df_orders.columns)
        if missing:
            raise ValueError(f"Missing required order columns: {sorted(missing)}")
        if "embedding" not in df_products.columns:
            raise ValueError("Product embeddings DataFrame must contain an 'embedding' column.")
        if category_col not in df_products.columns:
            raise ValueError(f"Product embeddings file must contain '{category_col}' column for category mix.")

        # 3) Merge (ürün embedding + kategori getir)
        df_products["StockCode"] = df_products["StockCode"].astype(str)
        df = (
            df_orders
            .merge(df_products[["StockCode", "embedding", category_col]], on="StockCode", how="inner")
            .dropna(subset=["embedding"])
        )
        if df.empty:
            raise ValueError("After merging, no rows with embeddings remain. Check inputs.")

        # 4) Zaman çürümesi
        decay_arr = half_life_decay(df["InvoiceDate"], half_life_days)  # np.ndarray
        decay = pd.Series(decay_arr, index=df.index)

        # 5) ROBUST WEIGHTING: log1p ve/veya cap (util.robust_qty_weight varsa kullan)
        try:
            from utils import robust_qty_weight  # opsiyonel
            weights = robust_qty_weight(df["Quantity"], decay.values, use_log1p=use_log1p, cap=qty_cap)
        except Exception:
            # inline fallback
            qty = df["Quantity"].astype(float).clip(lower=0)
            if use_log1p:
                qty = np.log1p(qty)
            if qty_cap is not None:
                qty = np.minimum(qty, float(qty_cap))
            weights = pd.Series(qty.values * decay.values, index=df.index)

        # 6) KATEGORİ CENTROID’LERİ: ürün bazında kategori ortalamaları
        #    (embedding’ler np.ndarray/list olabilir, hepsini np.array’e çeviriyoruz)
        cat_centroids: dict[str, np.ndarray] = {}
        for cat, g in df_products.dropna(subset=[category_col]).groupby(category_col):
            vecs = [np.asarray(v) for v in g["embedding"].dropna().tolist()]
            if len(vecs) == 0:
                continue
            cat_centroids[str(cat)] = np.vstack(vecs).mean(axis=0)

        # 7) Müşteri bazında hesaplama:
        #    - Ürün embedding’leri + robust weight ile vektör
        #    - Aynı anda kategori dağılımını toplayıp centroid karışımı ekle
        vectors: Dict[str, np.ndarray] = {}
        for cust_id, grp in df.groupby("CustomerID"):
            emb_stack = np.vstack(grp["embedding"].values)
            w = weights.loc[grp.index].values
            base_vec = weighted_average_stack(emb_stack, w)

            # kategori dağılımı (müşteri bazında)
            if 0.0 < cat_mix_alpha <= 0.5:
                # kategoriye göre ağırlık toplamları
                cat_w = grp.assign(w=w).groupby(category_col)["w"].sum()
                total_w = float(cat_w.sum()) or 1.0
                mix_vec = None
                for cat, cw in cat_w.items():
                    centroid = cat_centroids.get(str(cat))
                    if centroid is None:
                        continue
                    share = float(cw) / total_w  # p(category | customer)
                    if mix_vec is None:
                        mix_vec = share * centroid
                    else:
                        mix_vec += share * centroid
                if mix_vec is not None:
                    base_vec = (1.0 - cat_mix_alpha) * base_vec + (cat_mix_alpha) * mix_vec

            # normalize et
            if normalize:
                base_vec = unit_normalize(base_vec.reshape(1, -1)).ravel()

            vectors[cust_id] = base_vec

        out = pd.DataFrame({
            "CustomerID": list(vectors.keys()),
            "embedding": list(vectors.values()),
        })

        # 8) Kaydet (atomic overwrite)
        if output_path is None:
            output_path = getattr(self, "behavioral_output_path", None)
            if not output_path:
                base, ext = os.path.splitext(self.output_path)
                output_path = f"{base}_behavioral.pkl" if not ext else f"{base.replace(ext, '')}_behavioral{ext}"

        if output_path.endswith(os.sep) or (os.path.isdir(output_path) and not output_path.lower().endswith(".pkl")):
            output_path = os.path.join(output_path, "customer_behavioral_embeddings.pkl")

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        tmp_path = output_path + ".tmp"
        out.to_pickle(tmp_path)
        os.replace(tmp_path, output_path)
        print(f"✅ Behavioral embeddings (robust+cat-mix) saved to {output_path} (rows={len(out)})")

        # cache
        self._df_cust_emb = out.copy()
        return out


    def _load_products(self) -> pd.DataFrame:
        if self._df_products is None:
            self._df_products = pd.read_pickle(self.product_embedding_path)
        return self._df_products

    def _load_customer_embeddings(self) -> pd.DataFrame:
        # prefer cache (if generate_customer_embeddings was run in-session)
        if self._df_cust_emb is not None:
            return self._df_cust_emb
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(
                f"Customer embeddings file not found at {self.output_path}. "
                "Run generate_customer_embeddings() first."
            )
        self._df_cust_emb = pd.read_pickle(self.output_path)
        return self._df_cust_emb

    def _get_customer_matrix(self) -> tuple[np.ndarray, list[str]]:
        df_ce = self._load_customer_embeddings()
        # ensure embeddings are np arrays (in case they were stored as lists)
        vectors = [np.asarray(v) for v in df_ce["embedding"].tolist()]
        X = np.vstack(vectors)
        ids = df_ce["CustomerID"].astype(str).tolist()
        return X, ids

    # -----------------------------
    # Similar customers (cosine)
    # -----------------------------
    def find_similar_customers(self, customer_id: str, top_k: int = 10):
        """
        Return top_k most similar customers by cosine similarity.
        Excludes the query customer itself.
        """
        X, ids = self._get_customer_matrix()
        id_to_idx = {cid: i for i, cid in enumerate(ids)}
        if customer_id not in id_to_idx:
            raise ValueError(f"CustomerID {customer_id} not found in embeddings.")
        q_idx = id_to_idx[customer_id]
        scores = cosine_scores(X[q_idx], X)
        scores[q_idx] = -np.inf
        top_idx = topk_indices_desc(scores, top_k)
        return [(ids[i], float(scores[i])) for i in top_idx]

    # -----------------------------
    # Product recommendation (cosine)
    # -----------------------------
    def _unit_normalize(self, M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Row-wise L2 normalize to use dot as cosine."""
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms = np.clip(norms, eps, None)
        return M / norms

    def _get_customer_vector(self, customer_id: str) -> np.ndarray:
        X, ids = self._get_customer_matrix()
        id_to_idx = {cid: i for i, cid in enumerate(ids)}
        if customer_id not in id_to_idx:
            raise ValueError(f"CustomerID {customer_id} not found in embeddings.")
        return X[id_to_idx[customer_id]]

    def recommend_products_for_customer(self, customer_id: str, top_n: int = 10,
                                        exclude_recent_days: int = 180, only_new: bool = True) -> pd.DataFrame:
        """
        Recommend products by cosine similarity between customer vector and product embeddings.
        """
        # Load dependencies
        df_orders = self._load_orders().copy()
        df_prod = self._load_products().copy().dropna(subset=["embedding"])

        # Candidate product matrix
        prod_ids = df_prod["StockCode"].astype(str).tolist()
        P = np.vstack([np.asarray(e) for e in df_prod["embedding"].tolist()])

        # Customer vector
        c = self._get_customer_vector(customer_id)

        # Cosine scores via util
        scores = cosine_scores(c, P)

        # Novelty / recency filters
        exclude_mask = np.zeros(len(prod_ids), dtype=bool)
        hist = df_orders[df_orders["CustomerID"] == customer_id]

        if only_new:
            ever_bought = set(hist["StockCode"].unique())
            if ever_bought:
                idxs = [i for i, sc in enumerate(prod_ids) if sc in ever_bought]
                exclude_mask[idxs] = True
        elif exclude_recent_days > 0:
            cutoff = df_orders["InvoiceDate"].max() - pd.Timedelta(days=exclude_recent_days)
            recent = set(hist[hist["InvoiceDate"] >= cutoff]["StockCode"].unique())
            idxs = [i for i, sc in enumerate(prod_ids) if sc in recent]
            exclude_mask[idxs] = True

        # Apply mask then take top-N
        scores[exclude_mask] = -np.inf
        top_idx = topk_indices_desc(scores, min(top_n, len(scores)))

        rows = []
        desc_col = "Description" if "Description" in df_prod.columns else None
        for i in top_idx:
            if np.isneginf(scores[i]):
                continue
            rows.append({
                "StockCode": prod_ids[i],
                "Description": (df_prod.iloc[i][desc_col] if desc_col else None),
                "Score": float(scores[i]),
            })
        return pd.DataFrame(rows)