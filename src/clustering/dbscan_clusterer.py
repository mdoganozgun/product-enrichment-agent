"""
DBSCAN clustering over generic feature vectors.

Treats dense regions as clusters; labels noise as -1. Silhouette is computed
excluding noise points.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .base_clusterer import BaseClusterer
from . import logger
import os
import ast


class DBSCANClusterer(BaseClusterer):
    """
    Perform DBSCAN clustering on row-level feature vectors.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.model: DBSCAN | None = None
        self.labels_: np.ndarray | None = None
        self.silhouette: float | None = None

    def load_features(self, path: str = "../data/customer_features.pkl") -> None:
        """
        Load a feature table from CSV or PKL and normalize it to a standard schema.

        Rules:
        - Accepts either a Pickle file (.pkl) or CSV (.csv).
        - If a legacy column named 'embedding' is found, it is renamed to 'features'.
        - If 'features' column contains stringified lists (e.g., "[0.1, 0.2]"), it will be parsed.
        - If no 'features' column exists, all numeric columns are combined as the feature vector.
        - Rows with all-numeric features are kept; any NaN/Inf handling happens in cluster().

        Parameters
        ----------
        path : str
            Path to a .csv or .pkl file containing features.
        """
        logger.info(f"Loading features from: {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Features file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".pkl":
            df = pd.read_pickle(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError("Unsupported file extension. Use .csv or .pkl")

        # Legacy mapping: 'embedding' -> 'features'
        if "features" not in df.columns and "embedding" in df.columns:
            logger.info("Auto-mapping legacy 'embedding' -> 'features'.")
            df = df.rename(columns={"embedding": "features"})

        def _coerce_feature_row(obj):
            # Already a numpy array or list
            if isinstance(obj, (list, np.ndarray)):
                return np.asarray(obj, dtype=float)
            # Try to parse stringified list
            if isinstance(obj, str):
                try:
                    parsed = ast.literal_eval(obj)
                    return np.asarray(parsed, dtype=float)
                except Exception:
                    # Fall back: not a list-like string
                    return obj
            return obj

        if "features" in df.columns:
            # Coerce each row to a numeric numpy array
            df["features"] = df["features"].apply(_coerce_feature_row)
            # If still not arrays (e.g., objects), raise a helpful error
            bad_mask = ~df["features"].apply(lambda x: isinstance(x, np.ndarray))
            if bad_mask.any():
                sample = df.loc[bad_mask, "features"].iloc[0]
                raise ValueError(f"'features' column contains non-vector data (e.g., {type(sample)}). "
                                 f"Ensure each row is a list/array or provide numeric columns.")
        else:
            # No explicit 'features' column: use all numeric columns as features
            id_like = {"CustomerID", "StockCode", "cluster"}
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_like]
            if not numeric_cols:
                raise ValueError("No numeric columns found to build 'features'. Add a 'features' column or numeric inputs.")
            logger.info(f"No 'features' column found; using numeric columns as features: {numeric_cols}")
            df["features"] = df[numeric_cols].apply(lambda r: r.values.astype(float), axis=1)

        # Final sanity: stack to ensure consistent shapes
        try:
            _ = np.vstack(df["features"].values)
        except Exception as e:
            raise ValueError(f"Failed to stack feature vectors. Ensure consistent lengths. Root cause: {e}")

        self.df = df
        logger.debug(f"Features loaded. Shape: {self.df.shape}")

    def cluster(self, eps: float | None = None, min_samples: int | None = None, **_: dict) -> None:
        """
        Fit DBSCAN on features.

        Parameters
        ----------
        eps : float | None
            Neighborhood radius.
        min_samples : int | None
            Minimum points per core.
        """
        self._ensure_features_loaded()
        if eps is not None:
            self.eps = eps
        if min_samples is not None:
            self.min_samples = min_samples

        X = np.vstack(self.df["features"].values)
        # Drop rows that contain NaN/Inf in their feature vectors
        bad_row_mask = ~np.isfinite(X).all(axis=1)
        if bad_row_mask.any():
            dropped = int(bad_row_mask.sum())
            logger.warning(f"Dropping {dropped} rows with NaN/Inf values before DBSCAN.")
            # Keep only good rows
            self.df = self.df.loc[~bad_row_mask].reset_index(drop=True)
            X = X[~bad_row_mask]

        # Standardize features so all dimensions contribute comparably to distance
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        logger.info(f"Fitting DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.model.fit_predict(X)
        self.df["cluster"] = self.labels_
        noise = int(np.sum(self.labels_ == -1))
        logger.info(f"DBSCAN finished. Found {len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)} clusters; {noise} noise points.")

    def evaluate(self) -> None:
        """Compute Silhouette score excluding noise points (-1)."""
        self._ensure_features_loaded()
        if self.labels_ is None:
            raise ValueError("No labels found. Run `cluster()` first.")

        mask = self.labels_ != -1
        if mask.sum() < 2 or len(set(self.labels_[mask])) < 2:
            logger.warning("Not enough non-noise clusters for Silhouette score.")
            self.silhouette = None
            return

        X = np.vstack(self.df.loc[mask, "features"].values)
        self.silhouette = silhouette_score(X, self.labels_[mask])
        logger.info(f"Silhouette Score (DBSCAN, noise excluded): {self.silhouette:.4f}")

    def save_results(self, path: str = "../data/customer_segments.csv") -> None:
        """
        Save identifier and cluster label assignments to CSV.
        Noise points are labeled as -1.
        """
        self._ensure_features_loaded()
        id_col = "CustomerID" if "CustomerID" in self.df.columns else self.df.columns[0]
        out = self.df[[id_col, "cluster"]].copy()
        out.to_csv(path, index=False)
        logger.info(f"Customer segments saved to: {path}")