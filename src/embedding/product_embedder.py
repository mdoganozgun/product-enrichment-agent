import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
logger = logging.getLogger(__name__)

# This class is responsible for converting product metadata into semantic vectors using Sentence-BERT
class ProductEmbedder:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        # Load the sentence-transformer model
        self.model = SentenceTransformer(model_name)

    # Combine relevant text fields from a product row into one string
    @staticmethod
    def create_embedding_text(row):
        parts = []
        for col in ["Description", "category", "sub_category", "usage_context"]:
            val = row.get(col, "")
            if pd.notna(val):
                parts.append(str(val).strip())
        tags = row.get("tags", [])
        if isinstance(tags, list):
            tag_str = " ".join([str(tag).strip() for tag in tags if tag])
            parts.append(tag_str)
        return " ".join(parts)

    # Apply embedding to an entire dataframe of products
    def embed_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df["embedding_text"] = df.apply(self.create_embedding_text, axis=1)
        texts = df["embedding_text"].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        df["embedding"] = embeddings
        return df

    # Embed a single string (e.g., search query)
    def embed_text(self, text: str):
        return self.model.encode(text)

    # Get the embedding of a specific product by its ID from a given DataFrame
    def get_product_embedding(self, df: pd.DataFrame, product_id):
        row = df[df["StockCode"] == product_id]
        if row.empty:
            logger.warning(f"Product ID {product_id} not found in DataFrame.")
            return None
        if "embedding" not in row.columns:
            logger.error("Embedding column not found in DataFrame.")
            return None
        return row.iloc[0]["embedding"]