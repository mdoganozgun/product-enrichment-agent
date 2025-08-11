"""
Two-stage clustering: DBSCAN to remove noise, then KMeans on the filtered set.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from .base_clusterer import BaseClusterer
from . import logger


class FilteredKMeansClusterer(BaseClusterer):
    """
    Run DBSCAN for outlier filtering, then apply KMeans on the remaining points.
    """

    def __init__(self, dbscan_eps: float = 0.5, dbscan_min_samples: int = 5, kmeans_n_clusters: int = 3) -> None:
        super().__init__()
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.kmeans_n_clusters = kmeans_n_clusters

        self.dbscan_labels_ = None
        self.kmeans_labels_ = None
        self.silhouette_kmeans_: float | None = None
        self.df_filtered: pd.DataFrame | None = None

    def load_features(self, path: str = "../../data/customer_embeddings.pkl") -> None:
        logger.info(f"Loading features from: {path}")
        df = pd.read_pickle(path)
        if "features" not in df.columns and "embedding" in df.columns:
            logger.info("Auto-mapping legacy 'embedding' -> 'features'.")
            df = df.rename(columns={"embedding": "features"})
        if "features" not in df.columns:
            raise ValueError("Expected a 'features' column in the features file.")
        self.df = df
        logger.debug(f"Features loaded. Shape: {self.df.shape}")

    def cluster(
        self,
        dbscan_eps: float | None = None,
        dbscan_min_samples: int | None = None,
        kmeans_n_clusters: int | None = None,
        **_: dict,
    ) -> None:
        """
        Step 1: DBSCAN to mark noise (-1).
        Step 2: KMeans on the filtered (non-noise) subset.
        """
        self._ensure_features_loaded()

        if dbscan_eps is not None:
            self.dbscan_eps = dbscan_eps
        if dbscan_min_samples is not None:
            self.dbscan_min_samples = dbscan_min_samples
        if kmeans_n_clusters is not None:
            self.kmeans_n_clusters = kmeans_n_clusters

        # --- DBSCAN ---
        X = np.vstack(self.df["features"].values)
        logger.info(f"DBSCAN filtering (eps={self.dbscan_eps}, min_samples={self.dbscan_min_samples})...")
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        self.dbscan_labels_ = dbscan.fit_predict(X)
        self.df["dbscan_cluster"] = self.dbscan_labels_

        noise_count = int(np.sum(self.dbscan_labels_ == -1))
        logger.info(f"DBSCAN completed. Noise points: {noise_count}")

        # Filter out noise
        self.df_filtered = self.df[self.df["dbscan_cluster"] != -1].copy()
        if self.df_filtered.empty:
            logger.warning("All points were marked as noise. KMeans will be skipped.")
            self.kmeans_labels_ = None
            self.df["cluster"] = -1
            self.labels_ = self.df["cluster"].values
            return

        # --- KMeans ---
        X_filtered = np.vstack(self.df_filtered["features"].values)
        logger.info(f"KMeans on filtered set (k={self.kmeans_n_clusters}) with {len(X_filtered)} points...")
        kmeans = KMeans(n_clusters=self.kmeans_n_clusters, random_state=42)
        self.kmeans_labels_ = kmeans.fit_predict(X_filtered)

        # Write back only for non-noise points; keep noise = -1
        self.df_filtered["cluster"] = self.kmeans_labels_
        self.df["cluster"] = -1
        self.df.loc[self.df_filtered.index, "cluster"] = self.df_filtered["cluster"]

        # Final labels
        self.labels_ = self.df["cluster"].values

    def evaluate(self) -> None:
        """Compute Silhouette score for KMeans labels on non-noise subset."""
        self._ensure_features_loaded()
        if self.kmeans_labels_ is None or self.df_filtered is None or self.df_filtered.empty:
            logger.warning("No valid clusters to evaluate (all noise or clustering not run).")
            self.silhouette_kmeans_ = None
            return

        X_filtered = np.vstack(self.df_filtered["features"].values)
        if len(set(self.kmeans_labels_)) < 2:
            logger.warning("Not enough clusters for Silhouette score on filtered set.")
            self.silhouette_kmeans_ = None
            return

        self.silhouette_kmeans_ = silhouette_score(X_filtered, self.kmeans_labels_)
        logger.info(f"Silhouette Score (KMeans after DBSCAN): {self.silhouette_kmeans_:.4f}")

    def save_results(self, path: str = "../data/customer_segments.csv") -> None:
        """Save final clusters (including noise=-1) for all rows."""
        self._ensure_features_loaded()
        id_col = "CustomerID" if "CustomerID" in self.df.columns else self.df.columns[0]
        out = self.df[[id_col, "cluster"]].copy()
        out.to_csv(path, index=False)
        logger.info(f"Customer segments saved to: {path}")