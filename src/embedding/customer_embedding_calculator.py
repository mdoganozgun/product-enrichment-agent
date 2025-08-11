# embedding/customer_embedding_calculator.py

import numpy as np
import pandas as pd
from utils.customer_data_utils import get_customer_purchase_matrix


class CustomerEmbeddingCalculator:
    def __init__(self, product_embeddings_df):
        self.product_embeddings_df = product_embeddings_df

    def calculate_customer_embeddings(self, strategy="default", time_decay_lambda=None):
        """
        Calculates customer embeddings by aggregating product embeddings with optional weighting.

        Args:
            strategy (str): Weighting strategy. Options:
                - "default": Equal weights
                - "quantity": Weighted by purchase quantity
                - "price": Weighted by unit price
                - "time": Exponential decay by recency
                - "time_quantity_price": Combined weight of recency, quantity, and price
            time_decay_lambda (float): Decay rate for recency weighting

        Returns:
            pd.DataFrame: DataFrame with columns ["CustomerID", "embedding"]
        """
        purchase_matrix = get_customer_purchase_matrix(self.product_embeddings_df)
        customer_embeddings = []

        for customer_id, group in purchase_matrix.items():
            try:
                embeddings = np.stack(group["embedding"].values)
            except ValueError:
                continue  # Skip customers with no embeddings

            weights = self._compute_weights(group, strategy, time_decay_lambda)
            weights = weights / (weights.sum() + 1e-8)  # normalize

            aggregated = np.average(embeddings, axis=0, weights=weights)
            customer_embeddings.append({"CustomerID": customer_id, "embedding": aggregated})

        return pd.DataFrame(customer_embeddings)

    def _compute_weights(self, df, strategy, time_decay_lambda):
        n = len(df)
        if strategy == "default":
            return np.ones(n)
        elif strategy == "quantity":
            return df["Quantity"].values
        elif strategy == "price":
            return df["UnitPrice"].values
        elif strategy == "time":
            return self._compute_time_weights(df, time_decay_lambda)
        elif strategy == "time_quantity_price":
            time_w = self._compute_time_weights(df, time_decay_lambda)
            quantity = df["Quantity"].values
            price = df["UnitPrice"].values
            return time_w * quantity * price
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def _compute_time_weights(self, df, time_decay_lambda):
        if time_decay_lambda is None:
            return np.ones(len(df))
        max_time = df["InvoiceDate"].max()
        deltas = (max_time - df["InvoiceDate"]).dt.total_seconds()
        return np.exp(-time_decay_lambda * deltas)