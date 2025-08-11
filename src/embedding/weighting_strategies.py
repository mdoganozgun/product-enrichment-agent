

import numpy as np
import pandas as pd

def price_only_weights(group: pd.DataFrame, lambda_decay: float = 0.001) -> np.ndarray:
    return group["UnitPrice"].values

def time_price_weights(group: pd.DataFrame, lambda_decay: float = 0.001) -> np.ndarray:
    latest_date = group["InvoiceDate"].max()
    time_deltas = (latest_date - group["InvoiceDate"]).dt.days
    time_weights = np.exp(-lambda_decay * time_deltas)
    return group["UnitPrice"].values * time_weights

def time_price_quantity_weights(group: pd.DataFrame, lambda_decay: float = 0.001) -> np.ndarray:
    latest_date = group["InvoiceDate"].max()
    time_deltas = (latest_date - group["InvoiceDate"]).dt.days
    time_weights = np.exp(-lambda_decay * time_deltas)
    return group["UnitPrice"].values * group["Quantity"].values * time_weights

WEIGHTING_FUNCTIONS = {
    "price_only": price_only_weights,
    "time_price": time_price_weights,
    "time_price_quantity": time_price_quantity_weights,
}