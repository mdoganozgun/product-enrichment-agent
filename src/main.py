import numpy as np
import pandas as pd

from embedding.customer_embedder import CustomerEmbedder
from manager.financial_profile_generator import generate_customer_financial_profile
from clustering.clustering_pipeline import run_clustering_pipeline
from visualization.clustering_visualizer import plot_clusters_2d, plot_clusters_3d
from visualization.silhouette_analyzer import plot_silhouette_scores


def main():
    # Embedding step

    # embedder = CustomerEmbedder()
    # embedder.generate_customer_embeddings(
    #     time_decay_lambda=0.001
    # )

    generate_customer_financial_profile(
        input_path="../data/enriched_retail.csv",
        output_path="../data/customer_financial_reports.csv"
    )

    # Clustering step
    run_clustering_pipeline(
        method="kmeans",  # "kmeans" | "dbscan" | "filtered_kmeans"
        dbscan_eps=0.5,
        dbscan_min_samples=5,
        kmeans_n_clusters=3,
        features_path="../data/customer_financial_reports.csv",  # içinde 'features' (veya legacy 'embedding') olmalı
        output_path="../data/customer_segments.csv"
    )

    # Visualization step

    # df = pd.read_pickle("data/customer_clusters_filtered_kmeans.pkl")
    # embeddings = np.vstack(df["embedding"].values)
    # labels = df["cluster"].values

    from visualization.clustering_visualizer import plot_clusters_2d, plot_clusters_3d, plot_elbow_method

    # X: (n_samples, n_features) numpy array / DataFrame.values
    # y: (n_samples,) label vektörü (DBSCAN için noise=-1 olabilir)

    df_clusters = pd.read_csv("../data/customer_segments.csv")

    # Özellik matrisi
    X = df_clusters.select_dtypes(include=[np.number]).drop(columns=["cluster"]).values

    # Küme etiketleri
    y = df_clusters["cluster"].values

    plot_clusters_2d(X, y, title="KMeans (2D)", standardize=True, save_path="logs/figs/kmeans_2d.png")
    plot_clusters_3d(X, y, title="KMeans (3D)", standardize=True, save_path="logs/figs/kmeans_3d.png")

    # Elbow örneği
    scores = [1200, 980, 910, 905]  # inertia / WCSS
    plot_elbow_method(scores, ks=[2, 3, 4, 5], save_path="logs/figs/elbow.png")


    # plot_clusters_2d(embeddings, labels)
    # plot_clusters_3d()

    # Silhouette analysis
    # plot_silhouette_scores(range(2, 11))

if __name__ == "__main__":
    main()
