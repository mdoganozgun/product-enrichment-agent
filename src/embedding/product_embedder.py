


import pandas as pd
from sentence_transformers import SentenceTransformer

# This class is responsible for converting product metadata into semantic vectors using Sentence-BERT
class ProductEmbedder:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        # Load the sentence-transformer model
        self.model = SentenceTransformer(model_name)

    # Combine relevant text fields from a product row into one string
    def create_embedding_text(self, row):
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
        # Generate text to be embedded
        df["embedding_text"] = df.apply(self.create_embedding_text, axis=1)
        # Generate vector embeddings using the model
        df["embedding"] = df["embedding_text"].apply(lambda x: self.model.encode(x))
        return df

    # Embed a single string (e.g., search query)
    def embed_text(self, text: str):
        return self.model.encode(text)