import os
import logging
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class CustomerSegmenter:
    def __init__(self, feature_path="data/combined_features.csv", output_dir="data", log_dir="logs"):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.feature_path = feature_path
        self.output_path = os.path.join(output_dir, "customer_segments.csv")
        self.elbow_plot_path = os.path.join(output_dir, "elbow_plot.png")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "segmentation.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.df = None
        self.customer_ids = None
        self.X_scaled = None

    def load_and_preprocess_features(self):
        try:
            self.logger.info("Loading customer features from %s", self.feature_path)

            # 1. CSV'den veri oku
            df = pd.read_csv(self.feature_path)

            # 2. Eksik verileri tamamen kaldır (CustomerID dahil)
            df.dropna(inplace=True)
            self.logger.info("Dropped rows with missing values. Remaining rows: %d", df.shape[0])

            # 3. CustomerID'yi ayır
            self.customer_ids = df["CustomerID"]
            X = df.drop("CustomerID", axis=1)

            # 4. Kategorik sütunları label encoding ile sayısal hale getir
            for col in X.select_dtypes(include="object").columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            # 5. Normalize et
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            # 6. Outlier tespiti ve kaldırılması
            model = IsolationForest(contamination=0.05, random_state=0)
            preds = model.fit_predict(X_scaled)

            X_clean = X_scaled[preds == 1]
            self.customer_ids = self.customer_ids[preds == 1].reset_index(drop=True)

            # 7. Final dataframe oluştur
            self.df = pd.DataFrame(X_clean, columns=X.columns)
            self.df.insert(0, "CustomerID", self.customer_ids)

            self.X_scaled = X_clean
            self.logger.info("Features loaded, encoded, normalized, and outliers removed. Final shape: %s",
                             self.X_scaled.shape)

        except Exception as e:
            self.logger.error("Error loading or preprocessing features")
            self.logger.error(e)
            raise

    def plot_correlation_matrix(self, input_path="data/customer_segments.csv"):
        try:
            self.logger.info("Plotting correlation matrix...")

            df = pd.read_csv(input_path)
            df = df.drop(columns=["CustomerID", "Cluster"], errors='ignore')  # sayısal olmayanları çıkar
            corr = df.corr(numeric_only=True)

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig("data/correlation_matrix.png")
            plt.show()

            self.logger.info("Correlation matrix saved to data/correlation_matrix.png")
        except Exception as e:
            self.logger.error("Failed to plot correlation matrix")
            self.logger.error(e)

    def plot_elbow(self, max_k=10):
        inertia = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(self.X_scaled)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, max_k + 1), inertia, marker='o')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method For Optimal k")
        plt.grid(True)
        plt.savefig(self.elbow_plot_path)
        plt.close()
        self.logger.info("Elbow plot saved to %s", self.elbow_plot_path)

    def segment_customers(self, n_clusters=5):
        try:
            self.logger.info("Clustering customers with KMeans (k=%d)...", n_clusters)

            if self.df is None or self.X_scaled is None or self.customer_ids is None:
                raise ValueError("Features not loaded. Please run load_and_preprocess_features() first.")

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            clusters = kmeans.fit_predict(self.X_scaled)
            self.df["Cluster"] = clusters

            df_with_cluster = pd.concat([self.customer_ids, self.df.drop("CustomerID", axis=1)], axis=1)
            df_with_cluster.to_csv(self.output_path, index=False)

            self.logger.info("Customer segments saved to %s", self.output_path)
        except Exception as e:
            self.logger.error("Failed to perform clustering")
            self.logger.error(e)
            raise

    def visualize_clusters_2d(self, input_path="data/customer_segments.csv"):
        try:
            self.logger.info("Visualizing customer clusters...")
            df = pd.read_csv(input_path)
            X = df.drop(columns=["CustomerID", "Cluster"])

            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=df["Cluster"], cmap="tab10", s=50)
            plt.title("Customer Segments (PCA Projection)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.colorbar(scatter, label="Cluster")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            self.logger.info("Cluster visualization completed.")
        except Exception:
            self.logger.error("Failed to visualize clusters")
            self.logger.error(traceback.format_exc())

    def visualize_clusters_3d(self, input_path="data/customer_segments.csv"):
        try:
            self.logger.info("Generating 3D PCA visualization of clusters...")

            # Load segmented data
            df = pd.read_csv(input_path)
            X = df.drop(columns=["CustomerID", "Cluster"])

            # Apply PCA for 3D
            pca = PCA(n_components=3)
            X_reduced = pca.fit_transform(X)
            df_plot = pd.DataFrame(X_reduced, columns=["PC1", "PC2", "PC3"])
            df_plot["Cluster"] = df["Cluster"].astype(str)

            # Plot with Plotly
            fig = px.scatter_3d(df_plot, x="PC1", y="PC2", z="PC3", color="Cluster",
                                title="3D PCA - Customer Segments", opacity=0.8)
            fig.show()

            self.logger.info("3D PCA cluster plot rendered successfully.")
        except Exception:
            self.logger.error("Failed to generate 3D cluster visualization")
            self.logger.error(traceback.format_exc())

if __name__ == "__main__":
    segmenter = CustomerSegmenter()
    segmenter.load_and_preprocess_features()
    segmenter.plot_elbow()
    segmenter.segment_customers(n_clusters=4)
    segmenter.plot_correlation_matrix()
    segmenter.visualize_clusters_2d()
    segmenter.visualize_clusters_3d()