

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

def generate_customer_financial_profile(
    input_path: str = "../../data/enriched_retail.csv",
    output_path: str = "../../data/customer_financial_reports.csv"
) -> None:
    """
    Build customer-level financial features from a line-item retail dataset and save as CSV.

    Parameters
    ----------
    input_path : str
        Path to the cleaned/enriched retail CSV. Expected columns include at least:
        ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID'].
    output_path : str
        Path to write the resulting customer financial features CSV.

    Notes
    -----
    - Handles unified cancellation detection using BOTH rules:
        * InvoiceNo starting with 'C'   (credit notes)
        * Negative Quantity             (returns)
    - Adds favorite shopping hour (most frequent hour).
    - Monthly spending statistics are computed per customer (mean/std), with std filled to 0 when undefined.
    - Spending trend is computed as the slope of a linear regression over monthly spend (0 if insufficient data).
    - All numeric NaNs are imputed to 0 before saving to keep downstream clustering simple and robust.
    """

    # ---- Load & basic hygiene
    df = pd.read_csv(input_path, parse_dates=["InvoiceDate"])
    # Keep only rows with essential fields
    df.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"], inplace=True)

    # Ensure proper dtypes
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df.dropna(subset=["Quantity", "UnitPrice"], inplace=True)

    # Derived columns
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceDay"] = df["InvoiceDate"].dt.date
    df["Weekday"] = df["InvoiceDate"].dt.weekday
    df["Hour"] = df["InvoiceDate"].dt.hour

    # Unified cancellation flag: startswith('C') OR Quantity < 0
    cancel_by_prefix = df["InvoiceNo"].astype(str).str.startswith("C")
    cancel_by_qty = df["Quantity"] < 0
    df["Transaction_Status"] = np.where(cancel_by_prefix | cancel_by_qty, "Cancelled", "Completed")

    # Sort to ensure stable diffs
    df.sort_values(["CustomerID", "InvoiceDate"], inplace=True)

    profiles = []

    # Pre-compute a global most recent date for recency
    global_last_date = df["InvoiceDate"].max()

    # ---- Group by customer and compute features
    for cid, group in df.groupby("CustomerID", sort=False):
        group = group.sort_values("InvoiceDate")

        # Core counts and sums
        total_transactions = group["InvoiceNo"].nunique()
        total_products_purchased = group["Quantity"].sum()
        total_spend = group["TotalPrice"].sum()
        avg_transaction_value = (total_spend / total_transactions) if total_transactions else 0.0
        unique_products = group["StockCode"].nunique()

        # Recency & frequency (days between purchases based on distinct invoices)
        purchase_dates = group.drop_duplicates("InvoiceNo")["InvoiceDate"]
        days_between = purchase_dates.diff().dropna().dt.days
        avg_days_between = float(days_between.mean()) if not days_between.empty else 0.0

        last_purchase = purchase_dates.max()
        days_since_last = float((global_last_date - last_purchase).days) if pd.notnull(last_purchase) else 0.0

        # Preferences
        weekday_mode = group["Weekday"].mode()
        day_of_week = int(weekday_mode.iloc[0]) if not weekday_mode.empty else 0

        hour_mode = group["Hour"].mode()
        favorite_hour = int(hour_mode.iloc[0]) if not hour_mode.empty else 0

        # Cancellations (transaction-level)
        cancellations = group.loc[group["Transaction_Status"] == "Cancelled", "InvoiceNo"].nunique()
        cancellation_rate = (cancellations / total_transactions) if total_transactions else 0.0

        # Monthly spending stats
        group["Month"] = group["InvoiceDate"].dt.to_period("M")
        monthly_spend = group.groupby("Month", sort=True)["TotalPrice"].sum()

        monthly_mean = float(monthly_spend.mean()) if len(monthly_spend) else 0.0
        monthly_std = float(monthly_spend.std()) if len(monthly_spend) > 1 else 0.0  # NaN -> 0 for single period

        # Spending trend via linear regression on monthly totals
        if len(monthly_spend) > 1:
            # months as 0..N-1
            x = np.arange(len(monthly_spend)).reshape(-1, 1)
            y = monthly_spend.values.reshape(-1, 1)
            trend_model = LinearRegression().fit(x, y)
            trend = float(trend_model.coef_[0][0])
        else:
            trend = 0.0

        profiles.append({
            "CustomerID": cid,
            "Days_Since_Last_Purchase": days_since_last,
            "Total_Transactions": int(total_transactions),
            "Total_Products_Purchased": float(total_products_purchased),
            "Total_Spend": float(total_spend),
            "Average_Transaction_Value": float(avg_transaction_value),
            "Unique_Products_Purchased": int(unique_products),
            "Average_Days_Between_Purchases": float(avg_days_between),
            "Day_Of_Week": int(day_of_week),
            "Favorite_Hour": int(favorite_hour),
            "Cancellation_Frequency": int(cancellations),
            "Cancellation_Rate": float(cancellation_rate),
            "Monthly_Spending_Mean": float(monthly_mean),
            "Monthly_Spending_Std": float(monthly_std),
            "Spending_Trend": float(trend),
        })

    profile_df = pd.DataFrame(profiles)

    # Final numeric NaN guard (should be none, but safe)
    num_cols = profile_df.select_dtypes(include=[np.number]).columns
    profile_df[num_cols] = profile_df[num_cols].fillna(0)

    profile_df.to_csv(output_path, index=False)
    print(f"✅ Financial profiles saved to {output_path}")

if __name__ == "__main__":
    generate_customer_financial_profile()