import os

import numpy as np
import pandas as pd
import logging
import traceback
from agent import enrich_product_description
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler


class EnrichmentManager:
    def __init__(self, data_path, cache_path, log_dir="logs"):
        # Ensure necessary directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs("../../data", exist_ok=True)

        # Define key file paths
        self.data_path = data_path
        self.cache_path = cache_path
        self.progress_path = os.path.join(log_dir, "progress.log")  # Tracks the last processed index
        self.log_path = os.path.join(log_dir, "enrichment.log")     # Detailed logging

        # Configure logging to both file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize state
        self.df = None
        self.df_cache = pd.DataFrame()
        self.descriptions = []

        # Columns expected in the enrichment output
        self.expected_cols = [
            "Description", "category", "sub_category", "usage_context", "price_segment",
            "material_type", "target_gender", "target_age_group", "tags"
        ]

    def load_dataset(self):
        # Load and clean the input retail dataset
        try:
            self.df = pd.read_csv(self.data_path)

            # Remove rows missing key identifiers
            self.df = self.df.dropna(subset=["Description", "CustomerID"])
            self.df.drop_duplicates(inplace=True)

            # Remove anomalous StockCodes with 0–1 digits
            unique_stock_codes = self.df['StockCode'].unique()
            anomalous = [code for code in unique_stock_codes if sum(c.isdigit() for c in str(code)) in (0, 1)]
            self.df = self.df[~self.df['StockCode'].isin(anomalous)]

            # Remove service-related records
            self.df = self.df[~self.df['Description'].isin(["Next Day Carriage", "High Resolution Image"])]

            # Normalize text and remove zero-priced items
            self.df['Description'] = self.df['Description'].str.upper()
            self.df = self.df[self.df['UnitPrice'] > 0]

            # Reset index and extract descriptions
            self.df.reset_index(drop=True, inplace=True)
            self.descriptions = self.df["Description"].tolist()
            self.logger.info("Dataset loaded with %d rows", len(self.df))
        except Exception:
            self.logger.error("Failed to load and clean dataset")
            self.logger.error(traceback.format_exc())
            raise

    def load_cache(self):
        if os.path.exists(self.cache_path) and os.path.getsize(self.cache_path) > 0:
            self.df_cache = pd.read_csv(self.cache_path)

            # Ensure enrichment_id column exists and is integer
            if "enrichment_id" not in self.df_cache.columns:
                self.df_cache.insert(0, "enrichment_id", range(len(self.df_cache)))
            else:
                self.df_cache["enrichment_id"] = pd.to_numeric(self.df_cache["enrichment_id"], errors="coerce").fillna(
                    0).astype(int)

            self.df_cache.to_csv(self.cache_path, index=False)
            self.logger.info("Loaded cache with %d records", len(self.df_cache))
        else:
            self.logger.info("No cache found. Starting fresh.")

    def get_resume_index(self):
        # Resume from the last saved index (if available)
        if os.path.exists(self.progress_path):
            with open(self.progress_path) as f:
                return int(f.read().strip())
        return 0

    def update_progress(self, index):
        # Save the current progress index
        with open(self.progress_path, "w") as f:
            f.write(str(index))

    def enrich_all(self):
        # Get only the descriptions that haven't been enriched yet
        existing_desc = set(self.df_cache["Description"]) if not self.df_cache.empty else set()
        to_enrich = [desc for desc in self.descriptions if desc not in existing_desc]

        # Start enrichment ID from the max in cache
        enrichment_start_id = self.df_cache["enrichment_id"].max() + 1 if not self.df_cache.empty else 0

        for i, desc in enumerate(to_enrich):
            try:
                # Call enrichment agent
                enriched = enrich_product_description(desc)
                enriched_dict = enriched.model_dump()

                # Add original description and unique enrichment ID
                enriched_dict["Description"] = desc
                enriched_dict["enrichment_id"] = enrichment_start_id + i

                # Ensure correct column order for cache CSV
                cache_cols = ["enrichment_id", "Description", "category", "sub_category", "usage_context",
                              "price_segment", "material_type", "target_gender", "target_age_group", "tags"]

                df_enriched = pd.DataFrame([enriched_dict])[cache_cols]

                # Only write header if file doesn't exist or is empty
                write_header = not os.path.exists(self.cache_path) or os.path.getsize(self.cache_path) == 0

                df_enriched.to_csv(
                    self.cache_path,
                    mode='a',
                    header=write_header,
                    index=False
                )

                # Update progress tracking
                self.update_progress(i + 1)
                self.logger.info("Enriched [ID %d]: %s", enriched_dict["enrichment_id"], desc)

                # Log progress every 10 descriptions
                if (i + 1) % 10 == 0 or (i + 1) == len(to_enrich):
                    remaining = len(to_enrich) - (i + 1)
                    self.logger.info("Progress: %d/%d done — %d remaining",
                                     i + 1, len(to_enrich), remaining)
            except Exception:
                # Log and continue on error
                self.logger.error("Failed at [%d] - %s", i, desc)
                self.logger.error(traceback.format_exc())
                continue

    def merge_and_save(self, output_path="data/enriched_retail.csv"):
        # Final merge of enriched data with the main dataset
        try:
            self.df_cache = pd.read_csv(self.cache_path)
            self.df_cache = self.df_cache[self.expected_cols]

            # Left join on Description
            self.df = self.df.merge(self.df_cache, on="Description", how="left")

            # Propagate enrichment across all matching descriptions
            for col in self.expected_cols[1:]:
                self.df[col] = self.df.groupby("Description")[col].transform(
                    lambda x: x.ffill().bfill().infer_objects(copy=False)
                )

            # Save final output
            self.df.to_csv(output_path, index=False)
            self.logger.info("Final enriched dataset saved to %s", output_path)
        except Exception:
            self.logger.error("Error during final merge or save")
            self.logger.error(traceback.format_exc())

    def generate_rfm(self, output_path="data/rfm_scores.csv"):
        try:
            self.logger.info("Generating RFM scores...")

            df = self.df.copy()
            df.dropna(subset=["CustomerID", "InvoiceDate", "InvoiceNo", "Quantity", "UnitPrice"], inplace=True)
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

            reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

            rfm = df.groupby("CustomerID").agg({
                "InvoiceDate": lambda x: (reference_date - x.max()).days,
                "InvoiceNo": "nunique",
                "TotalPrice": "sum"
            }).rename(columns={
                "InvoiceDate": "Recency",
                "InvoiceNo": "Frequency",
                "TotalPrice": "Monetary"
            })

            rfm.reset_index(inplace=True)
            rfm.to_csv(output_path, index=False)
            self.logger.info("RFM scores saved to %s", output_path)

        except Exception:
            self.logger.error("Failed to generate RFM scores")
            self.logger.error(traceback.format_exc())
            raise

    def generate_spending_profile(self, output_path="data/spending_profile.csv"):
        try:
            self.logger.info("Generating customer spending profile features...")

            df = self.df.copy()
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
            df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek

            # Ana metrikler
            profile = df.groupby("CustomerID").agg({
                "InvoiceNo": "nunique",
                "Quantity": "sum",
                "TotalPrice": "sum",
                "category": pd.Series.nunique,
                "target_gender": lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown",
                "target_age_group": lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown",
                "DayOfWeek": lambda x: x.mode().iloc[0] if not x.mode().empty else -1
            }).rename(columns={
                "InvoiceNo": "TotalTransactions",
                "Quantity": "TotalUnitsPurchased",
                "TotalPrice": "TotalSpend",
                "category": "CategoryCount",
                "target_gender": "PreferredGender",
                "target_age_group": "PreferredAgeGroup",
                "DayOfWeek": "PreferredDayOfWeek"
            })

            # Sepet ve işlem ortalamaları
            profile["AvgTransactionValue"] = profile["TotalSpend"] / profile["TotalTransactions"]
            profile["AvgBasketSize"] = profile["TotalUnitsPurchased"] / profile["TotalTransactions"]

            # İptal bilgileri
            cancellation_df = df[df["Quantity"] < 0].groupby("CustomerID").agg({
                "InvoiceNo": "nunique"
            }).rename(columns={"InvoiceNo": "CancellationFrequency"})

            profile = profile.merge(cancellation_df, on="CustomerID", how="left")
            profile["CancellationFrequency"] = profile["CancellationFrequency"].fillna(0)
            profile["CancellationRate"] = profile["CancellationFrequency"] / profile["TotalTransactions"]

            # En çok ürün alınan kategori
            most_purchased = df.groupby(["CustomerID", "category"])["Quantity"].sum()
            most_purchased = most_purchased.reset_index()
            most_purchased = most_purchased.loc[most_purchased.groupby("CustomerID")["Quantity"].idxmax()]
            most_purchased = most_purchased[["CustomerID", "category"]].rename(
                columns={"category": "MostPurchasedCategory"})

            # En çok harcama yapılan kategori
            most_spent = df.groupby(["CustomerID", "category"])["TotalPrice"].sum()
            most_spent = most_spent.reset_index()
            most_spent = most_spent.loc[most_spent.groupby("CustomerID")["TotalPrice"].idxmax()]
            most_spent = most_spent[["CustomerID", "category"]].rename(columns={"category": "MostSpentCategory"})

            # Birleştir
            profile = profile.merge(most_purchased, on="CustomerID", how="left")
            profile = profile.merge(most_spent, on="CustomerID", how="left")

            # Gereksiz kolonları kaldır
            profile.drop(columns=["TotalUnitsPurchased"], inplace=True)

            profile.reset_index(drop=True, inplace=True)
            profile.to_csv(output_path, index=False)
            self.logger.info("Spending profile saved to '%s' with shape %s", output_path, profile.shape)

        except Exception:
            self.logger.error("Failed to generate spending profile")
            self.logger.error(traceback.format_exc())
            raise

    def combine_rfm_and_spending_profile(self,
                                         rfm_path="data/rfm_scores.csv",
                                         spending_path="data/spending_profile.csv",
                                         output_path="data/combined_features.csv"):
        try:
            self.logger.info("Combining RFM and Spending Profile features...")

            # RFM ve spending profile dosyalarını yükle
            rfm_df = pd.read_csv(rfm_path)
            spending_df = pd.read_csv(spending_path)

            # CustomerID üzerinden birleştir
            combined_df = pd.merge(rfm_df, spending_df, on="CustomerID", how="inner")

            # One-hot encoding yerine label encoding tercih edilebilir; burada bırakıyoruz
            combined_df.to_csv(output_path, index=False)

            self.logger.info("Combined features saved to '%s' with shape %s", output_path, combined_df.shape)

        except Exception:
            self.logger.error("Failed to combine RFM and spending profile features")
            self.logger.error(traceback.format_exc())
            raise



if __name__ == "__main__":
    manager = EnrichmentManager("../../data/Online Retail.csv", "data/enriched_cache.csv")
    manager.load_dataset()
    manager.load_cache()
    # # manager.enrich_all()
    manager.merge_and_save()
    manager.generate_rfm()
    manager.generate_spending_profile()
    manager.combine_rfm_and_spending_profile()
