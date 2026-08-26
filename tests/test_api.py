"""tests/test_api.py

Integration tests for the FastAPI service.
"""

from fastapi.testclient import TestClient
import pytest
from service.app import app


@pytest.fixture
def client():
  with TestClient(app) as c:
    yield c


def test_health_endpoint(client):
  response = client.get("/health")
  assert response.status_code == 200
  assert response.json()["status"] == "healthy"


def test_flight_prediction_endpoint(client):
  payload = {
      "airline": "IndiGo",
      "source_city": "Delhi",
      "destination_city": "Mumbai",
      "departure_time": "Morning",
      "arrival_time": "Afternoon",
      "stops": "zero",
      "class": "Economy",
      "duration_hours": 2.15,
      "days_left": 18,
      "current_listed_price": 5400.0,
  }

  response = client.post("/api/v1/flights/predict", json=payload)
  assert response.status_code == 200

  data = response.json()
  assert "predicted_fair_price_inr" in data
  assert data["predicted_fair_price_inr"] > 1000.0
  assert "advisory" in data
  assert data["advisory"]["recommendation"] in ["BUY_NOW", "WAIT", "FAIR_PRICE"]
  assert "latency_ms" in data