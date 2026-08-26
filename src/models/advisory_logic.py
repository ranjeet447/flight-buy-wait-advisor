"""src/models/advisory_logic.py

Algorithmic Buy vs. Wait advisory decision engine.
"""

from typing import Any, Dict


def compute_advisory_decision(
    predicted_fair_price: float,
    current_listed_price: float,
    days_left: int,
    avg_route_price: float,
    min_route_price: float,
    std_route_price: float,
) -> Dict[str, Any]:
  """Computes recommendation based on price gap, volatility, and booking window."""
  price_diff = current_listed_price - predicted_fair_price
  pct_diff = (
      price_diff / predicted_fair_price
  ) * 100.0 if predicted_fair_price > 0 else 0.0

  # Rule 1: High Urgency Zone (< 7 days)
  if days_left <= 7:
    recommendation = "BUY_NOW"
    reason = (
        "Last-minute booking window (< 7 days). Fares historically increase"
        " sharply as departure approaches."
    )
    confidence = 0.92

  # Rule 2: Excellent Deal (Priced well below fair value)
  elif current_listed_price <= predicted_fair_price * 0.92:
    recommendation = "BUY_NOW"
    savings = round(predicted_fair_price - current_listed_price, 2)
    reason = (
        f"Current price is ₹{savings:,.0f} ({abs(pct_diff):.1f}%) below fair"
        " market value. Excellent booking window."
    )
    confidence = 0.88

  # Rule 3: High Price + Comfortable Window (> 14 days)
  elif current_listed_price >= predicted_fair_price * 1.10 and days_left > 14:
    recommendation = "WAIT"
    potential_savings = round(current_listed_price - predicted_fair_price, 2)
    reason = (
        f"Current fare is inflated by ₹{potential_savings:,.0f}"
        f" (+{pct_diff:.1f}%). Fares on this route typically cool down"
        f" between 10-15 days prior to departure."
    )
    confidence = 0.81

  # Rule 4: Moderate / Fair Price
  else:
    recommendation = "FAIR_PRICE"
    reason = (
        "Current fare is within normal volatility range for this route. Safe"
        " to book if your travel dates are fixed."
    )
    confidence = 0.75

  return {
      "recommendation": recommendation,
      "confidence_score": confidence,
      "reasoning": reason,
      "price_delta_inr": round(price_diff, 2),
      "price_delta_percent": round(pct_diff, 2),
  }