import pandas as pd
import numpy as np
from typing import Dict
from collections import defaultdict

from embedding.weighting_strategies import WEIGHTING_FUNCTIONS
from src.utils.customer_data_utils import (
    load_customer_data,
    align_customer_product_embeddings,
    apply_time_price_quantity_weights
)


class CustomerEmbedder:
    """
    CustomerEmbedder is responsible for generating customer-level embedding vectors
    by aggregating product embeddings based on various weighting strategies.
    """

    def __init__(
        self,
        enriched_data_path: str = "../data/enriched_retail.csv",
        product_embedding_path: str = "../data/products_with_embeddings.pkl",
        output_path: str = "../data/customer_embeddings.pkl"
    ):
        self.enriched_data_path = enriched_data_path
        self.product_embedding_path = product_embedding_path
        self.output_path = output_path

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load enriched order data and product embeddings."""
        return load_customer_data(
            self.enriched_data_path,
            self.product_embedding_path
        )

    def align_customer_embeddings(self, df_orders: pd.DataFrame, df_products: pd.DataFrame) -> pd.DataFrame:
        """Merge orders and products on StockCode and remove rows with missing embeddings."""
        return align_customer_product_embeddings(df_orders, df_products)

    def apply_embedding_weights(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted importance of each product in customer baskets."""
        return apply_time_price_quantity_weights(merged_df)

    def save_embeddings(self, customer_vectors: Dict[int, np.ndarray]) -> None:
        df_out = pd.DataFrame({
            "CustomerID": list(customer_vectors.keys()),
            "embedding": list(customer_vectors.values())
        })
        df_out.to_pickle(self.output_path)
        print(f"✅ Customer embeddings saved to {self.output_path}")

    def generate_customer_embeddings(self, weighting_strategy: str = "price_only",
                                     time_decay_lambda: float = 0.001) -> None:
        """
        Main interface for generating customer embeddings.

        Parameters:
        - weighting_strategy (str): Strategy for calculating customer vectors.
            Options:
                * "price_only" - weights only by line total price.
                * "time_price" - adds exponential time decay to price.
                * "time_price_quantity" - combines time-decayed price and quantity.
        - time_decay_lambda (float): Decay factor for time-sensitive strategies.

        Output:
        - Pickle file of customer embeddings saved to self.output_path
        """
        df_orders, df_products = self.load_data()
        df_merged = self.align_customer_embeddings(df_orders, df_products)
        df_merged = self.apply_embedding_weights(df_merged)

        compute_fn = WEIGHTING_FUNCTIONS.get(weighting_strategy, WEIGHTING_FUNCTIONS["price_only"])

        customer_vectors = {}

        for customer_id, group in df_merged.groupby("CustomerID"):
            weights = compute_fn(group, lambda_decay=time_decay_lambda)
            embeddings = np.vstack(group["embedding"].values)
            customer_vector = np.average(embeddings, axis=0, weights=weights)
            customer_vectors[customer_id] = customer_vector

        if isinstance(customer_vectors, dict):
            self.save_embeddings(customer_vectors)
        else:
            raise TypeError("Expected customer_vectors to be a dictionary mapping CustomerID to embedding vectors, but got a different type. Please check the weighting function implementation.")