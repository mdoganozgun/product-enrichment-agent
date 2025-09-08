import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# EK: util'ten mevcut yardımcılar
from utils import load_transactions, kmeans_cluster  # CSV okuma + dosya var mı kontrolü

from embedding.product_embedder import ProductEmbedder
from embedding.milvus_client import MilvusClient

# This class orchestrates the embedding and Milvus operations
class EmbeddingManager:
    def __init__(self,
                 product_csv_path="../data/enriched_cache.csv",  # Path to the CSV file containing product data
                 embedding_pkl_path="../data/products_with_embeddings.pkl",  # Path where the embeddings will be saved/loaded
                 model_name="paraphrase-MiniLM-L6-v2",  # Name of the Sentence-BERT model to use
                 milvus_host="localhost",  # Hostname of the Milvus server
                 milvus_port="19530",  # Port of the Milvus server
                 collection_name="products"):  # Name of the Milvus collection
        # Save file paths and initialize internal components
        self.csv_path = product_csv_path  # CSV input path
        self.pkl_path = embedding_pkl_path  # Pickle output path for embeddings

        # Initialize the product embedding model
        self.embedder = ProductEmbedder(model_name)

        # Initialize the Milvus client for database operations
        self.milvus = MilvusClient(milvus_host, milvus_port, collection_name)

    def _read_embeddings_df(self) -> pd.DataFrame:
        """Read the cached product embeddings pickle."""
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f"Embeddings cache not found at: {self.pkl_path}. Run generate_product_embeddings() first.")
        df = pd.read_pickle(self.pkl_path)
        if "StockCode" not in df.columns or "embedding" not in df.columns:
            raise ValueError("Embeddings cache is missing required columns: 'StockCode' and/or 'embedding'.")
        return df

    def generate_product_embeddings(self):
        """
        Orchestrator: load CSV and delegate embedding+cache to ProductEmbedder.
        Note: any cleaning/prep is handled inside ProductEmbedder.
        """
        df = load_transactions(self.csv_path)

        # Delegate the full operation to the embedder (to be implemented there).
        return self.embedder.generate_and_cache_embeddings(df, self.pkl_path)

    # Search Milvus using embedded query text and return similar products
    def query_similar_products(self, query_text: str, top_k=5):
        vector = self.embedder.embed_text(query_text)
        return self.milvus.search_similar(vector, top_k)

    # Retrieve the embedding of a product from the saved pickle file
    def get_product_embedding(self, product_id):
        df = self._read_embeddings_df()
        match = df.loc[df["StockCode"].astype(str) == str(product_id), "embedding"]
        if match.empty:
            raise ValueError(f"Product {product_id} not found in embeddings cache.")
        return match.values[0]


    # Create Milvus collection (drops existing one)
    def create_milvus_collection(self):
        self.milvus.create_collection()
        print("✅ Milvus collection created.")

    # Insert embeddings from pickle into Milvus
    def insert_embeddings_to_milvus(self):
        df = self._read_embeddings_df()
        # ensure serializable lists
        emb_col = df["embedding"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)
        product_ids = df["StockCode"].astype(str).tolist()
        embeddings = emb_col.tolist()
        self.milvus.insert_embeddings(product_ids, embeddings)
        print("✅ Embeddings inserted into Milvus.")

    # Create index on the embedding field in Milvus (no logging).
    def create_index(self):
        """Create index on the embedding field in Milvus."""
        self.milvus.create_index()

    # Enrich raw Milvus search results with product descriptions
    def enrich_search_results(self, results):
        """
        Given a list of (StockCode, similarity score), return enriched data with product description.

        Args:
            results (List[Tuple[str, float]]): Milvus search results with product_id and distance.

        Returns:
            List[Dict]: Each dict contains StockCode, Description, and similarity Score.
        """
        df = self._read_embeddings_df()
        enriched = []
        for stockcode, score in results:
            row = df[df["StockCode"] == stockcode]
            if not row.empty:
                enriched.append({
                    "StockCode": stockcode,
                    "Description": row.iloc[0]["Description"],
                    "Score": score
                })
        return enriched

    def attach_behavioral_microsegments(
            self,
            reports_csv_path: str,
            behavioral_embeddings_pkl: str,
            out_csv_path: str | None = None,
            k: int | None = None,
            k_values: range | list[int] = range(2, 11),
            rel_drop_threshold: float = 0.1,
            return_metrics: bool = False,
            label_col_in_embeddings: str | None = None,
            plot: bool = True,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
        """
        Behavioral mikro-segment etiketlerini finansal rapora feature olarak ekler.
        - Eğer 'label_col_in_embeddings' verilmiş ve pkl içinde varsa, onu kullanır.
        - Aksi halde pkl içindeki 'embedding' ile util.kmeans_cluster üzerinden etiket üretir.
        - Sonucu rapora 'BehavioralCluster_...' kolonu olarak merge eder.
        """
        # 1) Dosyaları oku
        if not os.path.exists(reports_csv_path):
            raise FileNotFoundError(f"reports_csv_path not found: {reports_csv_path}")
        if not os.path.exists(behavioral_embeddings_pkl):
            raise FileNotFoundError(f"behavioral_embeddings_pkl not found: {behavioral_embeddings_pkl}")

        df_rep = load_transactions(reports_csv_path)
        if "CustomerID" not in df_rep.columns:
            raise ValueError("reports_csv must contain 'CustomerID' column.")

        df_beh = pd.read_pickle(behavioral_embeddings_pkl)
        if "CustomerID" not in df_beh.columns:
            raise ValueError("behavioral_embeddings_pkl must contain 'CustomerID' column.")

        X_for_plot = None
        # --- 2) Etiket kaynağı: ya hazır kolon ya da KMeans ile üret ---
        scan = None
        if label_col_in_embeddings and label_col_in_embeddings in df_beh.columns:
            col_name = label_col_in_embeddings
            labels = df_beh[col_name].astype(int).values
            sil = float("nan")
            if "embedding" in df_beh.columns:
                X_for_plot = np.vstack([np.asarray(v) for v in df_beh["embedding"].tolist()])
        else:
            if "embedding" not in df_beh.columns:
                raise ValueError("Embeddings file must contain an 'embedding' column when no label_col_in_embeddings is provided.")
            # Build matrix
            X = np.vstack([np.asarray(v) for v in df_beh["embedding"].tolist()])

            if k is None:
                k_grid = list(k_values) if isinstance(k_values, (list, range)) else list(k_values)
                silhouettes = []
                best_k = None
                best_s = -np.inf

                for k_try in k_grid:
                    try:
                        _, s = kmeans_cluster(X, int(k_try))
                        silhouettes.append(float(s) if not np.isnan(s) else np.nan)
                        if not np.isnan(s) and s > best_s:
                            best_s = s
                            best_k = int(k_try)
                    except Exception:
                        silhouettes.append(np.nan)
                        continue

                if best_k is None:
                    raise ValueError("Could not fit KMeans for any k in k_values; all silhouettes are NaN/invalid.")
                k = best_k
                scan = {"k_grid": k_grid, "silhouettes": silhouettes}

            labels, sil = kmeans_cluster(X, int(k))
            X_for_plot = X
            col_name = f"BehavioralCluster_KMeans_k{k}"

        # --- Ensure we don't duplicate the SAME column name on repeated runs ---
        # If the exact target column already exists in the report, drop it so merge writes a fresh one.
        if col_name in df_rep.columns:
            df_rep = df_rep.drop(columns=[col_name])

        # 3) Müşteri-etiket tablosu
        assign = pd.DataFrame({
            "CustomerID": df_beh["CustomerID"].astype(str).values,
            col_name: labels.astype(int)
        })

        # 4) Merge (rapordaki tüm müşteriler kalsın)
        df_rep["CustomerID"] = df_rep["CustomerID"].astype(str)
        df_out = df_rep.merge(assign, on="CustomerID", how="left")

        # Fill missing labels with -1 and enforce int type
        if col_name not in df_out.columns:
            # Should not happen since we just merged that column from `assign`,
            # but keep a safe fallback.
            df_out[col_name] = -1
        df_out[col_name] = df_out[col_name].fillna(-1).astype(int)

        # Tipi netle
        df_out[col_name] = df_out[col_name].astype(int)

        # --- Optional: plot KMeans result in 2D via PCA ---
        if plot and X_for_plot is not None:
            try:
                pca = PCA(n_components=2, random_state=42)
                Z = pca.fit_transform(X_for_plot)
                plt.figure(figsize=(8, 6))
                for lab in sorted(np.unique(labels)):
                    mask = (labels == lab)
                    plt.scatter(Z[mask, 0], Z[mask, 1], s=25, alpha=0.8, label=f"Cluster {lab}")
                plt.title(f"KMeans (k={int(k)}), silhouette={float(sil):.3f}")
                plt.xlabel("PCA 1")
                plt.ylabel("PCA 2")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
            except Exception:
                # Silently ignore plotting errors to keep the pipeline robust
                pass

        # 5) Kaydet
        if out_csv_path is None:
            out_csv_path = reports_csv_path  # üzerine yaz
        os.makedirs(os.path.dirname(os.path.abspath(out_csv_path)), exist_ok=True)
        df_out.to_csv(out_csv_path, index=False)

        if return_metrics:
            return df_out, {"k": int(k), "silhouette": float(sil), "scan": scan, "label_col": col_name}
        else:
            return df_out
