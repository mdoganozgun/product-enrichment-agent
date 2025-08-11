"""
High-level orchestration to run a chosen clustering method end-to-end (feature-agnostic).
"""

from clustering.filtered_kmeans_clusterer import FilteredKMeansClusterer
from clustering.kmeans_clusterer import KMeansClusterer
from clustering.dbscan_clusterer import DBSCANClusterer
from . import logger


def run_clustering_pipeline(
    method: str = "kmeans",
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    kmeans_n_clusters: int = 3,
    features_path: str = "../data/customer_embeddings.pkl",   # was embedding_path
    output_path: str = "../data/customer_segments.csv",
) -> None:
    """
    Run the selected clustering pipeline on generic 'features' and save results.

    Parameters
    ----------
    method : {"kmeans", "dbscan", "filtered_kmeans"}
        Which clustering approach to execute.
    dbscan_eps : float
        Neighborhood radius for DBSCAN.
    dbscan_min_samples : int
        Minimum number of samples per core point for DBSCAN.
    kmeans_n_clusters : int
        Number of clusters for KMeans.
    features_path : str
        Path to pickle containing ['features'] (legacy: ['embedding'] is auto-mapped).
    output_path : str
        Destination CSV for clustering results.
    """
    logger.info(f"Running clustering pipeline: method='{method}' on features='{features_path}'")

    if method == "kmeans":
        clusterer = KMeansClusterer(n_clusters=kmeans_n_clusters)
        clusterer.load_features(features_path)
        clusterer.cluster()
        clusterer.evaluate()
        clusterer.save_results(output_path)

    elif method == "dbscan":
        clusterer = DBSCANClusterer(eps=dbscan_eps, min_samples=dbscan_min_samples)
        clusterer.load_features(features_path)
        clusterer.cluster()
        clusterer.evaluate()
        clusterer.save_results(output_path)

    elif method == "filtered_kmeans":
        clusterer = FilteredKMeansClusterer(
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            kmeans_n_clusters=kmeans_n_clusters,
        )
        clusterer.load_features(features_path)
        clusterer.cluster()
        clusterer.evaluate()
        clusterer.save_results(output_path)

    else:
        logger.error(f"Invalid method: {method}")
        raise ValueError("Invalid clustering method. Choose from 'kmeans', 'dbscan', 'filtered_kmeans'.")

    logger.info("Clustering pipeline finished successfully.")