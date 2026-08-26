"""src/features/build_features.py

Feature engineering and dataset preparation with 2026 fare calibration.
"""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/raw/Clean_Dataset.csv")
PROCESSED_DIR = Path("data/processed")
FEATURE_STORE_DATA = Path("feature_store/data")

# 2026 Indian Domestic Aviation Inflation Index
INFLATION_FACTOR_2026 = 1.65


def engineer_features():
  PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
  FEATURE_STORE_DATA.mkdir(parents=True, exist_ok=True)

  print("⚙️ Loading raw dataset...")
  df = pd.read_csv(RAW_DATA_PATH)

  if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

  # Calibrate base price to current 2026 market rates
  print(
      f"📈 Calibrating historical fares with {INFLATION_FACTOR_2026}x inflation"
      " factor for 2026..."
  )
  df["price"] = (df["price"] * INFLATION_FACTOR_2026).round(2)

  # 1. Create Entity Identifier: route_id
  df["route_id"] = (
      df["source_city"] + "_" + df["destination_city"] + "_" + df["class"]
  )

  # 2. Compute Route-Level Aggregate Features for Feast
  print("📊 Computing route-level historical aggregates...")
  route_stats = (
      df.groupby("route_id")
      .agg(
          avg_route_price=("price", "mean"),
          min_route_price=("price", "min"),
          max_route_price=("price", "max"),
          std_route_price=("price", "std"),
          avg_duration=("duration", "mean"),
      )
      .reset_index()
  )

  route_stats["std_route_price"] = route_stats["std_route_price"].fillna(0.0)

  current_time = datetime.now(timezone.utc)
  route_stats["event_timestamp"] = current_time
  route_stats["created_timestamp"] = current_time

  # Save Feast Feature Table (Parquet)
  feast_parquet_path = FEATURE_STORE_DATA / "route_features.parquet"
  route_stats.to_parquet(feast_parquet_path, index=False)
  print(f"✅ Feast Parquet feature table saved to {feast_parquet_path}")

  # 3. Create Enriched Training Dataset
  print("🧪 Generating enriched training dataset...")
  enriched_df = df.merge(
      route_stats.drop(columns=["event_timestamp", "created_timestamp"]),
      on="route_id",
      how="left",
  )

  enriched_df["price_to_avg_ratio"] = (
      enriched_df["price"] / enriched_df["avg_route_price"]
  )
  enriched_df["event_timestamp"] = current_time

  training_data_path = PROCESSED_DIR / "training_data.parquet"
  enriched_df.to_parquet(training_data_path, index=False)
  print(f"✅ Training data saved to {training_data_path}")


if __name__ == "__main__":
  engineer_features()