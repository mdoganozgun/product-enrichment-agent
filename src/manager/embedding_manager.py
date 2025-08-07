import pandas as pd

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

    # Clean product data using filtering rules
    def clean_product_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows with missing 'Description' or 'CustomerID'
        df = df.dropna(subset=["Description", "CustomerID"])

        # Drop duplicate rows
        df.drop_duplicates(inplace=True)

        # Remove anomalous stock codes (e.g., those with 0 or 1 digits)
        unique_stock_codes = df['StockCode'].unique()
        anomalous = [code for code in unique_stock_codes if sum(c.isdigit() for c in str(code)) in (0, 1)]
        df = df[~df['StockCode'].isin(anomalous)]

        # Remove irrelevant service descriptions
        df = df[~df['Description'].isin(["Next Day Carriage", "High Resolution Image"])]

        # Convert descriptions to uppercase and remove zero-priced items
        df['Description'] = df['Description'].str.upper()
        df = df[df['UnitPrice'] > 0]

        # Reset index
        df.reset_index(drop=True, inplace=True)

        return df

    # Generate embeddings for products and save them to a pickle file
    def generate_product_embeddings(self):
        import os
        import logging
        os.makedirs(os.path.dirname(self.pkl_path), exist_ok=True)
        os.makedirs("../logs", exist_ok=True)
        # Setup logging with timestamped log file handler for detailed logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("../logs/embedding.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

        if os.path.exists(self.pkl_path):
            existing_df = pd.read_pickle(self.pkl_path)
            if "StockCode" not in existing_df.columns:
                logger.warning("⚠️ 'StockCode' column missing in existing embeddings. Skipping resume logic.")
                existing_df = pd.DataFrame()
                embedded_ids = set()
            else:
                embedded_ids = set(existing_df["StockCode"].astype(str))
                logger.info(f"🔁 Resuming from previous embedding with {len(embedded_ids)} products already embedded.")
        else:
            existing_df = pd.DataFrame()
            embedded_ids = set()

        logger.info("📥 Reading CSV and applying cleaning steps...")
        df = pd.read_csv(self.csv_path)  # Load original product data
        df = self.clean_product_dataframe(df)  # Apply cleaning steps
        # Take only unique products by StockCode
        df = df.drop_duplicates(subset=["StockCode"])
        logger.info(f"📊 Total cleaned rows: {len(df)}")
        before_dedup = len(df)
        df = df[~df["StockCode"].astype(str).isin(embedded_ids)]
        logger.info(f"🧹 Skipping already embedded products: {before_dedup - len(df)} skipped, {len(df)} remaining.")
        logger.info(f"🧼 Data cleaned. Remaining rows: {len(df)}")

        # Generate embedding text and embeddings
        df["embedding_text"] = df.apply(self.embedder.create_embedding_text, axis=1)
        logger.info("🧠 Generating embeddings...")
        df = df.reset_index(drop=True)  # Ensure index is clean for safe at[] assignment
        df["embedding"] = [None] * len(df)
        from tqdm import tqdm
        embeddings = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔄 Embedding products"):
            product_id = row.get("StockCode", f"row-{idx}")
            logger.info(f"Embedding product {idx + 1}/{len(df)} - StockCode: {product_id}")
            text = row["embedding_text"]
            embedding = self.embedder.model.encode(text)
            embeddings.append(embedding)
        df["embedding"] = embeddings
        # df["embedding"] = df["embedding_text"].apply(lambda x: self.embedder.model.encode(x))

        # Ensure 'StockCode' column exists before saving
        if "StockCode" not in df.columns:
            raise ValueError("❌ 'StockCode' column missing from input data. Cannot proceed.")

        if not existing_df.empty:
            df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=["StockCode"])

        # Save final DataFrame with all necessary columns
        df.to_pickle(self.pkl_path)
        logger.info(f"💾 Embeddings saved to: {self.pkl_path}")
        logger.info(f"🔎 Columns in final DataFrame: {df.columns.tolist()}")
        logger.info(f"✅ {len(df)} products embedded and saved to {self.pkl_path}")
        logger.info(f"📦 Total cache size now: {len(df.drop_duplicates(subset=['StockCode']))}")

    # Search Milvus using embedded query text and return similar products
    def query_similar_products(self, query_text: str, top_k=5):
        vector = self.embedder.embed_text(query_text)
        return self.milvus.search_similar(vector, top_k)

    # Retrieve the embedding of a product from the saved pickle file
    def get_product_embedding(self, product_id):
        df = pd.read_pickle(self.pkl_path)
        return df.loc[df["StockCode"].astype(str) == str(product_id), "embedding"].values[0]


    # Create Milvus collection (drops existing one)
    def create_milvus_collection(self):
        self.milvus.create_collection()
        print("✅ Milvus collection created.")

    # Insert embeddings from pickle into Milvus
    def insert_embeddings_to_milvus(self):
        df = pd.read_pickle(self.pkl_path)
        df["embedding"] = df["embedding"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)
        product_ids = df["StockCode"].astype(str).tolist()
        embeddings = df["embedding"].tolist()
        self.milvus.insert_embeddings(product_ids, embeddings)
        print("✅ Embeddings inserted into Milvus.")

    # Create index on the embedding field in Milvus
    def create_index(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("⚙️ Creating Milvus index on 'embedding' field...")
        self.milvus.create_index()
        logger.info("✅ Milvus index created.")

    # Enrich raw Milvus search results with product descriptions
    def enrich_search_results(self, results):
        """
        Given a list of (StockCode, similarity score), return enriched data with product description.

        Args:
            results (List[Tuple[str, float]]): Milvus search results with product_id and distance.

        Returns:
            List[Dict]: Each dict contains StockCode, Description, and similarity Score.
        """
        df = pd.read_pickle(self.pkl_path)
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