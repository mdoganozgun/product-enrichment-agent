import pandas as pd
import numpy as np
from collections import defaultdict

class CustomerEmbedder:
    def __init__(self,
                 enriched_data_path="../data/enriched_retail.csv",
                 product_embedding_path="../data/products_with_embeddings.pkl",
                 output_path="../data/customer_embeddings.pkl"):
        self.enriched_data_path = enriched_data_path
        self.product_embedding_path = product_embedding_path
        self.output_path = output_path

    def load_data(self):
        df_orders = pd.read_csv(self.enriched_data_path)
        df_products = pd.read_pickle(self.product_embedding_path)
        return df_orders, df_products

    def align_and_validate(self, df_orders, df_products):
        if "embedding" not in df_products.columns:
            raise ValueError("❌ Product embeddings missing!")
        df_products["StockCode"] = df_products["StockCode"].astype(str)
        df_orders["StockCode"] = df_orders["StockCode"].astype(str)
        df_orders["CustomerID"] = df_orders["CustomerID"].astype(str)
        return df_orders, df_products

    def merge_and_weight(self, df_orders, df_products):
        df_merged = df_orders.merge(df_products[["StockCode", "embedding"]], on="StockCode", how="inner")
        df_merged["TotalPrice"] = df_merged["Quantity"] * df_merged["UnitPrice"]
        df_merged = df_merged[df_merged["TotalPrice"] > 0]
        df_merged["TotalPrice"] = np.log1p(df_merged["TotalPrice"])
        return df_merged

    def compute_customer_embeddings(self, df_merged):
        customer_vectors = {}
        for customer_id, group in df_merged.groupby("CustomerID"):
            weights = group["TotalPrice"].values
            embeddings = np.vstack(group["embedding"].values)
            weighted_avg = np.average(embeddings, axis=0, weights=weights)
            customer_vectors[customer_id] = weighted_avg
        return customer_vectors

    def compute_customer_embeddings_advanced(self, df_merged, time_decay_lambda=0.001):
        customer_vectors = {}
        df_merged["InvoiceDate"] = pd.to_datetime(df_merged["InvoiceDate"])
        max_date = df_merged["InvoiceDate"].max()
        df_merged["DaysAgo"] = (max_date - df_merged["InvoiceDate"]).dt.days
        df_merged["TimeWeight"] = np.exp(-time_decay_lambda * df_merged["DaysAgo"])
        df_merged["WeightedPrice"] = df_merged["TotalPrice"] * df_merged["TimeWeight"]

        for customer_id, group in df_merged.groupby("CustomerID"):
            weights = group["WeightedPrice"].values
            embeddings = np.vstack(group["embedding"].values)
            if len(weights) > 0 and np.sum(weights) > 0:
                weighted_avg = np.average(embeddings, axis=0, weights=weights)
                customer_vectors[customer_id] = weighted_avg
        return customer_vectors

    def compute_customer_embeddings_weighted_all(self, df_merged, time_decay_lambda=0.001):
        customer_vectors = {}
        df_merged["InvoiceDate"] = pd.to_datetime(df_merged["InvoiceDate"])
        max_date = df_merged["InvoiceDate"].max()
        df_merged["DaysAgo"] = (max_date - df_merged["InvoiceDate"]).dt.days
        df_merged["TimeWeight"] = np.exp(-time_decay_lambda * df_merged["DaysAgo"])
        df_merged["EnhancedWeight"] = df_merged["TotalPrice"] * df_merged["TimeWeight"]

        for customer_id, group in df_merged.groupby("CustomerID"):
            weights = group["EnhancedWeight"].values
            embeddings = np.vstack(group["embedding"].values)
            if len(weights) > 0 and np.sum(weights) > 0:
                weighted_avg = np.average(embeddings, axis=0, weights=weights)
                customer_vectors[customer_id] = weighted_avg
        return customer_vectors

    def generate_customer_embeddings(self, weighting_strategy="price_only", time_decay_lambda=0.001):
        df_orders, df_products = self.load_data()
        df_orders, df_products = self.align_and_validate(df_orders, df_products)
        df_merged = self.merge_and_weight(df_orders, df_products)

        if weighting_strategy == "time_price":
            customer_vectors = self.compute_customer_embeddings_advanced(df_merged, time_decay_lambda)
        elif weighting_strategy == "time_price_quantity":
            customer_vectors = self.compute_customer_embeddings_weighted_all(df_merged, time_decay_lambda)
        else:
            customer_vectors = self.compute_customer_embeddings(df_merged)

        df_out = pd.DataFrame({
            "CustomerID": list(customer_vectors.keys()),
            "embedding": list(customer_vectors.values())
        })
        df_out.to_pickle(self.output_path)
        print(f"✅ Customer embeddings saved to {self.output_path}")