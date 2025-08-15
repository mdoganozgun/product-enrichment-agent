import os
import pandas as pd
import numpy as np
from typing import Dict

from sklearn.metrics.pairwise import cosine_similarity


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
        enriched_data_path: str = "../data/enriched_retail.csv",
        product_embedding_path: str = "../data/products_with_embeddings.pkl",
        output_path: str = "../data/customer_embeddings.pkl",
    ):
        self.enriched_data_path = enriched_data_path
        self.product_embedding_path = product_embedding_path
        self.output_path = output_path

    # -----------------------------
    # Core API
    # -----------------------------
    def generate_customer_embeddings(
        self,
        half_life_days: int = 365,
        use_unit_price: bool = True,
    ) -> None:
        """
        Build customer embeddings with a time-decayed weighted average.

        Parameters
        ----------
        half_life_days : int, default=365
            Half-life in days for exponential decay. Example: with 365 days,
            a purchase from 1 year ago counts as half.
        use_unit_price : bool, default=True
            If True, multiply effective quantities by UnitPrice so that spend
            matters. If False, use only time-decayed quantities.
        """
        # 1) Load data
        df_orders = pd.read_csv(self.enriched_data_path)
        df_products = pd.read_pickle(self.product_embedding_path)

        # 2) Validate & align schema
        required_orders = {"StockCode", "CustomerID", "Quantity", "InvoiceDate"}
        if use_unit_price:
            required_orders.add("UnitPrice")
        missing_orders = required_orders - set(df_orders.columns)
        if missing_orders:
            raise ValueError(f"Missing required order columns: {sorted(missing_orders)}")

        if "embedding" not in df_products.columns:
            raise ValueError("Product embeddings DataFrame must contain an 'embedding' column.")

        # Cast to suitable types
        df_orders["StockCode"] = df_orders["StockCode"].astype(str)
        df_products["StockCode"] = df_products["StockCode"].astype(str)
        df_orders["CustomerID"] = df_orders["CustomerID"].astype(str)
        df_orders["InvoiceDate"] = pd.to_datetime(df_orders["InvoiceDate"], errors="coerce")
        df_orders = df_orders.dropna(subset=["InvoiceDate"]).copy()

        # 3) Merge orders with product embeddings
        df = df_orders.merge(df_products[["StockCode", "embedding"]], on="StockCode", how="inner")
        df = df.dropna(subset=["embedding"]).copy()
        if df.empty:
            raise ValueError("After merging, no rows with embeddings remain. Check inputs.")

        # 4) Time decay (half-life)
        max_date = df["InvoiceDate"].max()
        days_since = (max_date - df["InvoiceDate"]).dt.days.clip(lower=0)
        decay = np.power(0.5, days_since / float(half_life_days))

        # 5) Effective weights
        effective_qty = df["Quantity"].astype(float) * decay
        # Optional: ignore negative/zero weights caused by returns or zeros
        if use_unit_price:
            raw_weight = effective_qty * df["UnitPrice"].astype(float)
        else:
            raw_weight = effective_qty
        weights = np.clip(raw_weight, a_min=0.0, a_max=None)

        # 6) Aggregate per customer: weighted average
        customer_vectors: Dict[str, np.ndarray] = {}
        for cust_id, grp in df.groupby("CustomerID"):
            emb_stack = np.vstack(grp["embedding"].values)
            w = weights.loc[grp.index].values
            if np.all(w == 0):
                # Fallback: unweighted average if all weights vanish
                vec = emb_stack.mean(axis=0)
            else:
                vec = np.average(emb_stack, axis=0, weights=w)
            customer_vectors[cust_id] = vec

        # 7) Persist
        out_df = pd.DataFrame({
            "CustomerID": list(customer_vectors.keys()),
            "embedding": list(customer_vectors.values()),
        })

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out_df.to_pickle(self.output_path)
        print(f"✅ Customer embeddings saved to {self.output_path}")


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
    def find_similar_customers(
        self,
        customer_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Return top_k most similar customers by cosine similarity.
        Excludes the query customer itself.
        """
        X, ids = self._get_customer_matrix()
        id_to_idx = {cid: i for i, cid in enumerate(ids)}
        if customer_id not in id_to_idx:
            raise ValueError(f"CustomerID {customer_id} not found in embeddings.")

        q_idx = id_to_idx[customer_id]
        q_vec = X[q_idx:q_idx+1]  # shape (1, d)

        # cosine similarity vs all
        sims = cosine_similarity(q_vec, X)[0]  # (n,)
        # drop self
        sims[q_idx] = -np.inf

        # top-k indices
        top_idx = np.argpartition(-sims, kth=min(top_k, len(sims)-1))[:top_k]
        # sort by score desc
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results = [(ids[i], float(sims[i])) for i in top_idx]
        return results

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

    def recommend_products_for_customer(
        self,
        customer_id: str,
        top_n: int = 10,
        exclude_recent_days: int = 90,
        only_new: bool = True,
    ) -> pd.DataFrame:
        """
        Recommend products by cosine similarity between customer vector and product embeddings.

        Parameters
        ----------
        customer_id : str
            Target customer.
        top_n : int
            Number of products to return.
        exclude_recent_days : int
            Exclude products bought within the last N days (novelty bias).
            Set to 0 to disable date-based exclusion.
        only_new : bool
            If True, exclude ALL products that the customer has ever bought.
            If False, only exclude 'recent' (within exclude_recent_days).
        """
        df_orders = self._load_orders().copy()
        df_orders["CustomerID"] = df_orders["CustomerID"].astype(str)
        df_orders["StockCode"] = df_orders["StockCode"].astype(str)
        df_orders["InvoiceDate"] = pd.to_datetime(df_orders["InvoiceDate"], errors="coerce")
        df_orders = df_orders.dropna(subset=["InvoiceDate"])

        df_prod = self._load_products().copy()
        df_prod["StockCode"] = df_prod["StockCode"].astype(str)
        df_prod = df_prod.dropna(subset=["embedding"])

        # candidate products matrix
        prod_ids = df_prod["StockCode"].tolist()
        P = np.vstack([np.asarray(e) for e in df_prod["embedding"].tolist()])
        Pn = self._unit_normalize(P)

        # customer vector
        c = self._get_customer_vector(customer_id).reshape(1, -1)
        cn = self._unit_normalize(c)

        # cosine scores
        scores = (cn @ Pn.T).ravel()  # shape (num_products,)

        # novelty filters
        cust_hist = df_orders[df_orders["CustomerID"] == customer_id]
        exclude_mask = np.zeros(len(prod_ids), dtype=bool)

        if only_new:
            ever_bought = set(cust_hist["StockCode"].unique())
            if ever_bought:
                bought_idx = [i for i, sc in enumerate(prod_ids) if sc in ever_bought]
                exclude_mask[bought_idx] = True
        elif exclude_recent_days > 0:
            max_date = df_orders["InvoiceDate"].max()
            cutoff = max_date - pd.Timedelta(days=exclude_recent_days)
            recent = set(cust_hist[cust_hist["InvoiceDate"] >= cutoff]["StockCode"].unique())
            recent_idx = [i for i, sc in enumerate(prod_ids) if sc in recent]
            exclude_mask[recent_idx] = True

        # apply mask
        scores_masked = scores.copy()
        scores_masked[exclude_mask] = -np.inf

        # top-n
        if top_n > len(scores_masked):
            top_n = len(scores_masked)
        top_idx = np.argpartition(-scores_masked, kth=top_n-1)[:top_n]
        top_idx = top_idx[np.argsort(-scores_masked[top_idx])]

        rows = []
        # attach optional description if exists
        desc_col = "Description" if "Description" in df_prod.columns else None
        for i in top_idx:
            if np.isneginf(scores_masked[i]):
                continue
            rows.append({
                "StockCode": prod_ids[i],
                "Description": (df_prod.iloc[i][desc_col] if desc_col else None),
                "Score": float(scores[i]),
            })

        return pd.DataFrame(rows)