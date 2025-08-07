import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import os
from sklearn.metrics import silhouette_score

def load_customer_embeddings(path="../data/customer_embeddings.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Customer embeddings file not found at {path}")
    df = pd.read_pickle(path)
    if "CustomerID" not in df.columns or "embedding" not in df.columns:
        raise ValueError("Customer embeddings file must contain 'CustomerID' and 'embedding' columns.")
    return df

def normalize_embeddings(df):
    df["embedding"] = df["embedding"].apply(lambda x: normalize([x])[0])
    return df

def cluster_customers(df, n_clusters=5):
    embeddings = np.vstack(df["embedding"].values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)
    return df, kmeans

def visualize_clusters(df, title="Customer Clusters (PCA Projection)"):
    embeddings = np.vstack(df["embedding"].values)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    df["pca1"] = reduced[:, 0]
    df["pca2"] = reduced[:, 1]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df["pca1"], df["pca2"], c=df["cluster"], cmap="tab10", alpha=0.6)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_clusters(df, output_path="../data/customer_segments.csv"):
    df[["CustomerID", "cluster"]].to_csv(output_path, index=False)
    print(f"✅ Customer segments saved to: {output_path}")


# CustomerClustering class for programmatic use
class CustomerClustering:
    def __init__(self, embedding_path="../data/customer_embeddings.pkl"):
        self.embedding_path = embedding_path
        self.df = None
        self.kmeans_model = None

    def load_embeddings(self):
        self.df = load_customer_embeddings(self.embedding_path)
        self.df = normalize_embeddings(self.df)

    def run_clustering(self, n_clusters=5):
        self.df, self.kmeans_model = cluster_customers(self.df, n_clusters)

    def save_results(self, output_path="../data/customer_segments.csv"):
        save_clusters(self.df, output_path)

    def visualize(self, title="Customer Clusters (PCA Projection)"):
        visualize_clusters(self.df, title)

    def plot_elbow_method(self, max_k=10):
        embeddings = np.vstack(self.df["embedding"].values)
        inertias = []

        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.title("Elbow Method - Inertia vs. K")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.show()

    def plot_silhouette_scores(self, k_range=range(2, 11)):
        from sklearn.metrics import silhouette_score

        embeddings = np.vstack(self.df["embedding"].values)
        scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            scores.append(score)
            print(f"K={k} → Silhouette Score: {score:.4f}")

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), scores, marker='o')
        plt.title("Silhouette Scores for Different K Values")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_dbscan(self, eps=0.5, min_samples=5):
        embeddings = np.vstack(self.df["embedding"].values)

        # DBSCAN uygulama
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(embeddings)
        self.df["cluster"] = labels

        # Gürültü noktaları -1 olarak etiketlenir
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"🔍 DBSCAN: {n_clusters} cluster found, {n_noise} noise points")

        # Silhouette skoru sadece -1 olmayanlar için
        mask = labels != -1
        if len(set(labels[mask])) > 1:
            score = silhouette_score(embeddings[mask], labels[mask])
            print(f"📈 Silhouette Score (excluding noise): {score:.4f}")
        else:
            print("⚠️ Silhouette score not available (too few clusters)")

        return self.df

    def plot_dbscan_clusters(self, title="DBSCAN Clustering"):
        from mpl_toolkits.mplot3d import Axes3D

        embeddings = np.vstack(self.df["embedding"].values)
        reduced_2d = PCA(n_components=2).fit_transform(embeddings)

        self.df["pca1"] = reduced_2d[:, 0]
        self.df["pca2"] = reduced_2d[:, 1]

        unique_labels = sorted(self.df["cluster"].unique())
        colors = plt.cm.get_cmap("tab10", len(unique_labels))

        # 2D Plot
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(unique_labels):
            subset = self.df[self.df["cluster"] == label]
            label_name = "Noise" if label == -1 else f"Cluster {label}"
            plt.scatter(
                subset["pca1"],
                subset["pca2"],
                c=[colors(i)],
                label=label_name,
                alpha=0.6,
                edgecolors="k",
                linewidths=0.3
            )
        plt.title(f"{title} (2D PCA)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(title="Cluster Label", loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_filtered_kmeans_after_dbscan(self, dbscan_eps=0.5, dbscan_min_samples=5, kmeans_n_clusters=3):
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import numpy as np

        # Embedding vektörlerini çıkar
        embeddings = np.vstack(self.df["embedding"].values)

        # DBSCAN ile kümeleme
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean')
        dbscan_labels = dbscan.fit_predict(embeddings)
        self.df["dbscan_cluster"] = dbscan_labels

        # Gürültü (noise) olmayanları filtrele
        filtered_df = self.df[self.df["dbscan_cluster"] != -1].copy()
        print(f"🧹 {len(self.df) - len(filtered_df)} noise points removed by DBSCAN.")

        # Filtrelenmiş vektörleri al
        filtered_embeddings = np.vstack(filtered_df["embedding"].values)

        # KMeans uygulama
        kmeans = KMeans(n_clusters=kmeans_n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(filtered_embeddings)
        filtered_df["kmeans_cluster"] = kmeans_labels

        # Silhouette skoru
        score = silhouette_score(filtered_embeddings, kmeans_labels)
        print(f"📈 KMeans Silhouette Score after DBSCAN noise removal: {score:.4f}")

        # PCA ile görselleştirme
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(filtered_embeddings)
        filtered_df["pca1"] = reduced[:, 0]
        filtered_df["pca2"] = reduced[:, 1]

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            filtered_df["pca1"],
            filtered_df["pca2"],
            c=filtered_df["kmeans_cluster"],
            cmap="tab10",
            alpha=0.6
        )
        plt.title("KMeans after DBSCAN Noise Removal")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return filtered_df

    def run_clustering_pipeline(self, method="kmeans", n_clusters=5, dbscan_eps=0.5, dbscan_min_samples=5):
        """
        method: "kmeans", "dbscan", or "filtered_kmeans"
        """
        if method == "kmeans":
            self.run_clustering(n_clusters=n_clusters)
            self.compute_silhouette_score()
            self.visualize()
            self.save_results()

        elif method == "dbscan":
            self.run_dbscan(eps=dbscan_eps, min_samples=dbscan_min_samples)
            self.plot_dbscan_clusters()
            self.save_results()

        elif method == "filtered_kmeans":
            filtered_df = self.run_filtered_kmeans_after_dbscan(
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
                kmeans_n_clusters=n_clusters
            )
            # Save filtered_df results
            filtered_df[["CustomerID", "kmeans_cluster"]].rename(
                columns={"kmeans_cluster": "cluster"}
            ).to_csv("../data/customer_segments.csv", index=False)
            print("✅ Filtered KMeans segments saved to: ../data/customer_segments.csv")

        else:
            raise ValueError("Invalid clustering method. Choose from 'kmeans', 'dbscan', or 'filtered_kmeans'.")

def main():
    clusterer = CustomerClustering()
    clusterer.load_embeddings()
    clusterer.run_clustering_pipeline(method="filtered_kmeans", n_clusters=3)

if __name__ == "__main__":
    main()