import pandas as pd
import numpy as np

def load_customer_data(order_path: str, product_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load order and product data from specified paths.

    Args:
        order_path (str): Path to the CSV file containing customer-product orders.
        product_path (str): Path to the pickle file containing product embeddings.

    Returns:
        tuple: (orders DataFrame, product embeddings DataFrame)
    """
    orders_df = pd.read_csv(order_path)
    products_df = pd.read_pickle(product_path)
    return orders_df, products_df

def align_customer_product_embeddings(
    customer_df: pd.DataFrame,
    product_embeddings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge customer orders with product embeddings based on 'StockCode'.
    Only the 'embedding' column is merged from the product embeddings.
    Drop rows where embedding is missing.

    Args:
        customer_df (pd.DataFrame): DataFrame with customer orders.
        product_embeddings_df (pd.DataFrame): DataFrame with product embeddings (must contain 'StockCode' and 'embedding').

    Returns:
        pd.DataFrame: Merged DataFrame with aligned embeddings.
    """
    embedding_only_df = product_embeddings_df[["StockCode", "embedding"]]
    merged_df = pd.merge(customer_df, embedding_only_df, on="StockCode", how="inner")
    merged_df = merged_df.dropna(subset=["embedding"])
    return merged_df

def apply_time_price_quantity_weights(
    df: pd.DataFrame,
    lambda_time: float = 0.001
) -> pd.DataFrame:
    """
    Calculate time-decay weighted importance of each product in customer baskets.

    Args:
        df (pd.DataFrame): DataFrame with at least 'InvoiceDate', 'UnitPrice', and 'Quantity' columns.
        lambda_time (float, optional): Decay rate for time weighting. Defaults to 0.001.

    Returns:
        pd.DataFrame: DataFrame with added weight columns.
    """
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    max_date = df["InvoiceDate"].max()
    df["days_since"] = (max_date - df["InvoiceDate"]).dt.days
    df["time_weight"] = np.exp(-lambda_time * df["days_since"])
    df["price_weight"] = df["UnitPrice"]
    df["quantity_weight"] = df["Quantity"]
    df["total_weight"] = df["time_weight"] * df["price_weight"] * df["quantity_weight"]
    return df
def get_customer_purchase_matrix(
    merged_df: pd.DataFrame,
    customer_id_col: str = "CustomerID",
    embedding_col: str = "embedding"
) -> dict:
    """
    Organize product embeddings into customer-specific lists.

    Args:
        merged_df (pd.DataFrame): DataFrame containing customer and embedding info.
        customer_id_col (str): Name of the customer ID column.
        embedding_col (str): Name of the column containing product embeddings.

    Returns:
        dict: Dictionary mapping each customer ID to a list of product embeddings.
    """
    customer_embeddings = {}
    for customer_id, group in merged_df.groupby(customer_id_col):
        embeddings = group[embedding_col].tolist()
        customer_embeddings[customer_id] = embeddings
    return customer_embeddings