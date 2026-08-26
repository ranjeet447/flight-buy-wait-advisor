"""service/schemas.py

Pydantic request and response schemas for the FastAPI inference service.
"""

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class FlightPredictionRequest(BaseModel):
  airline: Literal[
      "IndiGo", "Air_India", "Vistara", "SpiceJet", "AirAsia", "GO_FIRST"
  ] = Field(..., example="IndiGo")
  source_city: Literal[
      "Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"
  ] = Field(..., example="Delhi")
  destination_city: Literal[
      "Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"
  ] = Field(..., example="Mumbai")
  departure_time: Literal[
      "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
  ] = Field(..., example="Morning")
  arrival_time: Literal[
      "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
  ] = Field(..., example="Afternoon")
  stops: Literal["zero", "one", "two_or_more"] = Field(..., example="zero")
  travel_class: Literal["Economy", "Business"] = Field(
      ..., alias="class", example="Economy"
  )
  duration_hours: float = Field(..., gt=0, example=2.15)
  days_left: int = Field(..., ge=1, le=60, example=18)
  current_listed_price: Optional[float] = Field(
      None,
      description="Current price shown on booking portal in INR",
      example=5200.0,
  )


class FlightPredictionResponse(BaseModel):
  route: str
  predicted_fair_price_inr: float
  expected_price_range: Dict[str, float]
  advisory: Dict[str, Any]
  features_enriched_from_feast: Dict[str, float]
  model_version: str
  latency_ms: float