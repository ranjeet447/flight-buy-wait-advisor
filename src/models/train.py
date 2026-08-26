"""src/models/train.py

Model training, evaluation, and experiment tracking with MLflow.
"""

import json
from pathlib import Path
import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = Path("data/processed/training_data.parquet")
MODEL_DIR = Path("models")


def train():
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  print("🚀 Loading training dataset...")
  df = pd.read_parquet(PROCESSED_DATA_PATH)

  cat_cols = [
      "airline",
      "source_city",
      "departure_time",
      "stops",
      "arrival_time",
      "destination_city",
      "class",
      "route_id",
  ]
  num_cols = [
      "duration",
      "days_left",
      "avg_route_price",
      "min_route_price",
      "max_route_price",
      "std_route_price",
      "avg_duration",
  ]

  for col in cat_cols:
    df[col] = df[col].astype("category")

  feature_cols = cat_cols + num_cols
  X = df[feature_cols]
  y = df["price"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  mlflow.set_experiment("indian_domestic_flight_advisor")

  hyperparams = {
      "objective": "regression",
      "metric": "mape",
      "boosting_type": "gbdt",
      "n_estimators": 500,
      "learning_rate": 0.05,
      "num_leaves": 63,
      "random_state": 42,
      "n_jobs": -1,
  }

  print("🧠 Training LightGBM model & logging to MLflow...")
  with mlflow.start_run(run_name="lightgbm_production_v2"):
    mlflow.log_params(hyperparams)

    model = lgb.LGBMRegressor(**hyperparams)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    mape = float(mean_absolute_percentage_error(y_test, preds)) * 100
    r2 = float(r2_score(y_test, preds))

    metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape_percent": round(mape, 2),
        "r2_score": round(r2, 4),
    }

    print("\n📈 Evaluation Metrics:")
    for k, v in metrics.items():
      print(f"  {k}: {v}")

    mlflow.log_metrics(metrics)

    with open("models/metrics.json", "w") as f:
      json.dump(metrics, f, indent=2)

    # 1. Save standalone decoupled model artifact for container serving
    joblib.dump(model, "models/model.joblib")
    print("💾 Standalone model saved to models/model.joblib")

    # 2. Register in MLflow Model Registry
    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path="flight_pricing_model",
        registered_model_name="IndianFlightPriceAdvisor",
    )
    print("✅ Model registered in MLflow Model Registry successfully!")


if __name__ == "__main__":
  train()