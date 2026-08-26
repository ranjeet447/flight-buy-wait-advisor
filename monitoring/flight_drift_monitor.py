"""monitoring/flight_drift_monitor.py

Evidently AI Data & Concept Drift Monitoring Suite.
"""

from pathlib import Path
try:
  # Evidently >= 0.7.0
  from evidently import Report
  from evidently.presets import DataDriftPreset
  EVIDENTLY_V2 = True
except ImportError:
  try:
    # Evidently >= 0.7.0 legacy path or 0.4.x-0.6.x
    from evidently.legacy.report import Report  # type: ignore
    from evidently.legacy.metric_preset import DataDriftPreset  # type: ignore
    EVIDENTLY_V2 = False
  except ImportError:
    from evidently.report import Report  # type: ignore
    from evidently.metric_preset import DataDriftPreset  # type: ignore
    EVIDENTLY_V2 = False
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REPORT_DIR = Path("monitoring/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_buckets=10):
  """Calculates Population Stability Index (PSI) between reference and current distributions."""
  ref_pct, bins = np.histogram(reference, bins=num_buckets)
  curr_pct, _ = np.histogram(current, bins=bins)

  ref_pct = np.where(ref_pct == 0, 1, ref_pct) / len(reference)
  curr_pct = np.where(curr_pct == 0, 1, curr_pct) / len(current)

  psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
  return float(psi_value)


def generate_drift_report():
  print("🔍 Loading Reference Training Data and Production Inference Stream...")

  ref_df = pd.read_parquet("data/processed/training_data.parquet")

  # Simulate recent production traffic with a 15% festival season fare surge
  prod_df = ref_df.sample(n=1000, random_state=42).copy()
  prod_df["price"] = (
      prod_df["price"] * np.random.uniform(1.08, 1.25, size=len(prod_df))
  ).round(2)
  prod_df["days_left"] = np.clip(
      prod_df["days_left"] - np.random.randint(1, 5, size=len(prod_df)), 1, 50
  )

  features_to_monitor = [
      "airline",
      "source_city",
      "destination_city",
      "departure_time",
      "stops",
      "class",
      "duration",
      "days_left",
      "price",
  ]

  reference = ref_df[features_to_monitor]
  current = prod_df[features_to_monitor]

  print("\n📊 Statistical Drift Analysis (K-S Test & PSI):")
  print("=" * 60)
  for col in ["duration", "days_left", "price"]:
    ks_stat, p_val = ks_2samp(reference[col], current[col])
    psi = calculate_psi(reference[col].values, current[col].values)
    status = (
        "🚨 DRIFT DETECTED"
        if psi > 0.10 or p_val < 0.05
        else "✅ IN CONTROL"
    )
    print(
        f"Feature: {col:<12} | PSI: {psi:.4f} | K-S p-val: {p_val:.4e} |"
        f" {status}"
    )
  print("=" * 60)

  print("\n📈 Generating interactive Evidently AI HTML dashboard...")
  report_path = REPORT_DIR / "flight_drift_report.html"
  if EVIDENTLY_V2:
    report = Report([DataDriftPreset()])
    eval_result = report.run(current_data=current, reference_data=reference)
    eval_result.save_html(str(report_path))
  else:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(report_path))
  print(f"✅ Live Drift Report generated: {report_path.resolve()}")


if __name__ == "__main__":
  generate_drift_report()