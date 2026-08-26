"""service/app.py

Production FastAPI inference service with Feast online enrichment.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import mlflow
import mlflow.lightgbm
import pandas as pd
from service.schemas import FlightPredictionRequest, FlightPredictionResponse
from src.models.advisory_logic import compute_advisory_decision

MODEL_LOCAL_PATH = Path("models/model.joblib")
ml_models = {}


def load_flight_model():
  if "model" in ml_models:
    return ml_models["model"]

  # 1. Primary: Load decoupled standalone artifact
  if MODEL_LOCAL_PATH.exists():
    print(f"🔄 Loading model from standalone artifact {MODEL_LOCAL_PATH}...")
    ml_models["model"] = joblib.load(MODEL_LOCAL_PATH)
    print("✅ Model loaded successfully from local artifact!")
    return ml_models["model"]

  # 2. Secondary: Fallback to MLflow Registry
  print("🔄 Loading LightGBM model from MLflow Registry...")
  try:
    ml_models["model"] = mlflow.lightgbm.load_model(
        "models:/IndianFlightPriceAdvisor/2"
    )
    print("✅ Model loaded successfully from registry!")
  except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")
  return ml_models["model"]


@asynccontextmanager
async def lifespan(app: FastAPI):
  load_flight_model()
  yield
  ml_models.clear()


app = FastAPI(
    title="Indian Domestic Flight 'Buy vs. Wait' Advisor API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
  return {"status": "healthy", "service": "flight-advisor-api"}


@app.post("/api/v1/flights/predict", response_model=FlightPredictionResponse)
def predict_flight_fare(payload: FlightPredictionRequest):
  start_time = time.time()
  model = load_flight_model()

  route_id = (
      f"{payload.source_city}_{payload.destination_city}_{payload.travel_class}"
  )

  try:
    from feast import FeatureStore

    store = FeatureStore(repo_path="feature_store")
    feature_vector = store.get_online_features(
        features=[
            "route_features:avg_route_price",
            "route_features:min_route_price",
            "route_features:max_route_price",
            "route_features:std_route_price",
            "route_features:avg_duration",
        ],
        entity_rows=[{"route_id": route_id}],
    ).to_dict()

    avg_p = float(feature_vector["avg_route_price"][0])
    min_p = float(feature_vector["min_route_price"][0])
    max_p = float(feature_vector["max_route_price"][0])
    std_p = float(feature_vector["std_route_price"][0])
    avg_d = float(feature_vector["avg_duration"][0])
  except Exception:
    stats_df = pd.read_parquet("feature_store/data/route_features.parquet")
    route_match = stats_df[stats_df["route_id"] == route_id]
    if not route_match.empty:
      row = route_match.iloc[0]
      avg_p, min_p, max_p, std_p, avg_d = (
          float(row["avg_route_price"]),
          float(row["min_route_price"]),
          float(row["max_route_price"]),
          float(row["std_route_price"]),
          float(row["avg_duration"]),
      )
    else:
      avg_p, min_p, max_p, std_p, avg_d = (
          8362.0,
          3147.0,
          43138.0,
          5224.0,
          2.5,
      )

  input_data = pd.DataFrame([{
      "airline": payload.airline,
      "source_city": payload.source_city,
      "departure_time": payload.departure_time,
      "stops": payload.stops,
      "arrival_time": payload.arrival_time,
      "destination_city": payload.destination_city,
      "class": payload.travel_class,
      "route_id": route_id,
      "duration": float(payload.duration_hours),
      "days_left": int(payload.days_left),
      "avg_route_price": float(avg_p),
      "min_route_price": float(min_p),
      "max_route_price": float(max_p),
      "std_route_price": float(std_p),
      "avg_duration": float(avg_d),
  }])

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
  for c in cat_cols:
    input_data[c] = input_data[c].astype("category")

  predicted_price = float(model.predict(input_data)[0])
  predicted_price = max(1000.0, round(predicted_price, 2))

  listed_price = (
      payload.current_listed_price
      if payload.current_listed_price is not None
      else predicted_price
  )
  advisory = compute_advisory_decision(
      predicted_fair_price=predicted_price,
      current_listed_price=listed_price,
      days_left=payload.days_left,
      avg_route_price=avg_p,
      min_route_price=min_p,
      std_route_price=std_p,
  )

  latency = (time.time() - start_time) * 1000.0

  return {
      "route": f"{payload.source_city} ➔ {payload.destination_city} ({payload.travel_class})",
      "predicted_fair_price_inr": predicted_price,
      "expected_price_range": {
          "min_expected_inr": round(max(min_p, predicted_price * 0.90), 2),
          "max_expected_inr": round(predicted_price * 1.12, 2),
      },
      "advisory": advisory,
      "features_enriched_from_feast": {
          "avg_route_price": round(avg_p, 2),
          "min_route_price": round(min_p, 2),
          "max_route_price": round(max_p, 2),
          "std_route_price": round(std_p, 2),
      },
      "model_version": "v2.0 (LightGBM)",
      "latency_ms": round(latency, 2),
  }