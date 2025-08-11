"""
KMeans clustering over generic feature vectors.

Loads features (CSV/PKL), fits KMeans, evaluates with Silhouette, and saves results.
Auto-detects input format and parses feature columns robustly.

Usage:
    clusterer = KMeansClusterer(n_clusters=3)
    clusterer.load_features("../data/customer_embeddings.pkl")   # or .csv
    clusterer.cluster()
    clusterer.evaluate()
    clusterer.save_results("../data/customer_segmenbu değişts.csv")
"""

from __future__ import annotations

import os
import ast
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .base_clusterer import BaseClusterer
from . import logger


class KMeansClusterer(BaseClusterer):
    """
    Perform KMeans clustering on row-level feature vectors.

    This class is file-format aware:
    - If you pass a `.csv` path to `load_features`, it will route to `load_features_csv`.
    - Otherwise it will route to `load_features_pkl`.

    Expected columns (at least one must exist):
    - 'features' : iterable of floats per row (preferred)
    - 'embedding': legacy name, will be auto-mapped to 'features'
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        """
        Initialize a KMeansClusterer with number of clusters and RNG seed.

        Parameters
        ----------
        n_clusters : int
            Number of KMeans clusters to learn.
        random_state : int
            Random seed for KMeans reproducibility.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model: KMeans | None = None
        self.silhouette: float | None = None

    # ---------- Public loading API (auto-dispatch) ----------

    def load_features(self, path: str) -> None:
        """
        Load features from a CSV or PKL file into `self.df`.

        The loader is selected by file extension:
        - `.csv` -> `load_features_csv`
        - otherwise -> `load_features_pkl`

        Parameters
        ----------
        path : str
            Path to a CSV/PKL with an array-like 'features' column
            (or legacy 'embedding', which will be remapped).
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            self.load_features_csv(path)
        else:
            self.load_features_pkl(path)

    # ---------- Specialized loaders ----------

    def load_features_pkl(self, path: str) -> None:
        """
        Load features from a pickle file into `self.df`.

        Parameters
        ----------
        path : str
            Path to pickle with at least ['CustomerID', 'features'] (or legacy ['embedding']).
        """
        logger.info(f"Loading PKL features from: {path}")
        df = pd.read_pickle(path)
        df = self._standardize_feature_column(df)
        # Ensure all entries are numpy arrays, then drop rows whose feature vectors contain NaNs
        df["features"] = df["features"].apply(self._as_numpy_vector)
        df = self._dropna_feature_rows(df)
        self.df = df
        logger.debug(f"PKL features loaded. Shape: {self.df.shape}")

    def load_features_csv(self, path: str) -> None:
        """
        Load features from a CSV file into `self.df`.

        Automatically parses string-encoded vectors like:
        - JSON lists: "[0.1, 0.2, 0.3]"
        - Python lists (literal): "[0.1, 0.2]"
        - Space/comma separated: "0.1 0.2 0.3" or "0.1,0.2,0.3"

        Parameters
        ----------
        path : str
            Path to CSV with 'features' or 'embedding' column.
        """
        logger.info(f"Loading CSV features from: {path}")
        df = pd.read_csv(path)
        df = self._standardize_feature_column(df)

        # Parse if features are strings
        if df["features"].apply(lambda x: isinstance(x, str)).any():
            logger.info("Parsing string-encoded feature vectors in CSV...")
            df["features"] = df["features"].apply(self._parse_vector_string)

        # Ensure all entries are numpy arrays
        df["features"] = df["features"].apply(self._as_numpy_vector)
        # Drop rows whose feature vectors contain NaNs
        df = self._dropna_feature_rows(df)
        self.df = df
        logger.debug(f"CSV features loaded. Shape: {self.df.shape}")

    # ---------- Core clustering pipeline ----------

    def cluster(self, n_clusters: int | None = None, **_: dict) -> None:
        """
        Fit KMeans on features.

        Parameters
        ----------
        n_clusters : int | None
            Optional override for number of clusters.
        """
        self._ensure_features_loaded()
        if n_clusters is not None:
            self.n_clusters = n_clusters

        X = self.get_feature_matrix()
        logger.info(f"Fitting KMeans (k={self.n_clusters}) on {len(X)} rows...")
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels_ = self.model.fit_predict(X)
        self.df["cluster"] = self.labels_
        logger.debug("KMeans clustering completed and labels assigned.")

    def evaluate(self) -> None:
        """
        Compute Silhouette score for current KMeans labels.

        Sets `self.silhouette` to None if scoring is not applicable
        (e.g., only one cluster present).
        """
        self._ensure_features_loaded()
        if self.labels_ is None:
            raise ValueError("No labels found. Run `cluster()` first.")

        X = self.get_feature_matrix()
        if len(set(self.labels_)) < 2:
            logger.warning("Not enough distinct clusters for Silhouette score.")
            self.silhouette = None
            return

        self.silhouette = silhouette_score(X, self.labels_)
        logger.info(f"Silhouette Score (KMeans): {self.silhouette:.4f}")

    def save_results(self, path: str = "../data/customer_segments.csv") -> None:
        """
        Save identifier and cluster label assignments to CSV.

        Parameters
        ----------
        path : str
            Output CSV path.
        """
        self._ensure_features_loaded()
        id_col = "CustomerID" if "CustomerID" in self.df.columns else self.df.columns[0]
        out = self.df[[id_col, "cluster"]].copy()
        out.to_csv(path, index=False)
        logger.info(f"Customer segments saved to: {path}")

    # ---------- Helpers ----------

    def _dropna_feature_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows whose 'features' vector contains any NaN values.

        Logs how many rows were removed.
        """
        before = len(df)
        # Convert to numpy and check for NaNs element-wise
        mask_ok = ~df["features"].apply(lambda v: np.isnan(self._as_numpy_vector(v)).any())
        df = df[mask_ok].copy()
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Dropped {removed} rows with NaNs in features.")
        return df

    def _standardize_feature_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame has a 'features' column.

        Behavior:
        - If 'embedding' exists and 'features' doesn't, rename it to 'features'.
        - If neither exists, automatically build a per-row feature vector by stacking
          numeric columns (excluding obvious identifier columns like *ID, InvoiceNo, etc.).
        - Logs which columns are used when auto-building.
        """
        # Case 1: legacy column mapping
        if "features" not in df.columns and "embedding" in df.columns:
            logger.info("Auto-mapping legacy 'embedding' -> 'features'.")
            df = df.rename(columns={"embedding": "features"})

        # Case 2: auto-build features from numeric columns
        if "features" not in df.columns:
            import numpy as _np

            # Identify candidate numeric columns
            numeric_cols = df.select_dtypes(include=[_np.number]).columns.tolist()

            # Exclude common identifier / key columns
            def _is_identifier(col: str) -> bool:
                lc = col.lower()
                return (
                        lc.endswith("id")
                        or lc == "id"
                        or lc in {"customerid", "stockcode", "invoice", "invoiceno", "orderid"}
                )

            feature_cols = [c for c in numeric_cols if not _is_identifier(c)]

            if not feature_cols:
                raise ValueError(
                    "Could not auto-build 'features': no suitable numeric columns found. "
                    "Provide a 'features' column (or legacy 'embedding')."
                )

            logger.info(
                "Auto-building 'features' from numeric columns: %s",
                ", ".join(feature_cols)
            )

            # Build a 1D float vector per row from selected numeric columns (row-wise)
            df["features"] = df[feature_cols].apply(lambda r: r.values.astype(float), axis=1)

        return df

    def _parse_vector_string(self, s: str) -> np.ndarray:
        """
        Parse a string representation of a vector into a numpy array.

        Supports:
        - JSON or Python list strings: "[0.1, 0.2]"
        - Space/comma separated numbers: "0.1 0.2 0.3" or "0.1,0.2,0.3"
        """
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return np.array([], dtype=float)

        s = s.strip()

        # Try JSON / literal list
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                return np.array(json.loads(s), dtype=float)
            except Exception:
                try:
                    return np.array(ast.literal_eval(s), dtype=float)
                except Exception:
                    pass

        # Fallback: split by comma or whitespace
        if "," in s:
            parts = s.split(",")
        else:
            parts = s.split()

        try:
            return np.array([float(x) for x in parts if str(x).strip() != ""], dtype=float)
        except Exception:
            logger.error(f"Failed to parse feature string: {s[:80]}...")
            raise

    def _as_numpy_vector(self, v: Any) -> np.ndarray:
        """
        Convert a value to a 1D numpy array of floats.
        """
        if isinstance(v, np.ndarray):
            return v.astype(float)
        if isinstance(v, (list, tuple)):
            return np.array(v, dtype=float)
        if isinstance(v, str):
            return self._parse_vector_string(v)
        # Unknown type: best effort
        return np.array(v, dtype=float).ravel()

    def get_feature_matrix(self) -> np.ndarray:
        """
        Stack the 'features' column into a 2D numpy array (n_samples, n_features).
        """
        self._ensure_features_loaded()
        return np.vstack(self.df["features"].apply(self._as_numpy_vector).values)