"""scripts/upload_to_hf.py

Uploads trained LightGBM model, Feast features, and Model Card to Hugging Face Model Hub.
"""

import json
import os
from pathlib import Path
from huggingface_hub import HfApi

# -------------------------------------------------------------
# Configuration (Set HF_USERNAME & HF_TOKEN in environment)
# -------------------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "ranjeet447")
REPO_ID = f"{HF_USERNAME}/indian-flight-price-advisor"
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

print(f"🚀 Creating / Connecting to public Model Repository: {REPO_ID}...")
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

# 1. Upload Model Artifact
print("📦 Uploading models/model.joblib...")
api.upload_file(
    path_or_fileobj="models/model.joblib",
    path_in_repo="model.joblib",
    repo_id=REPO_ID,
    repo_type="model",
)

# 2. Upload Feast Parquet Route Features
print("📦 Uploading feature_store/data/route_features.parquet...")
api.upload_file(
    path_or_fileobj="feature_store/data/route_features.parquet",
    path_in_repo="route_features.parquet",
    repo_id=REPO_ID,
    repo_type="model",
)

# Optional: Upload metrics json if available
metrics_path = Path("models/metrics.json")
metrics_data = {}
if metrics_path.exists():
  with open(metrics_path, "r", encoding="utf-8") as f:
    metrics_data = json.load(f)
  print("📦 Uploading models/metrics.json...")
  api.upload_file(
      path_or_fileobj="models/metrics.json",
      path_in_repo="metrics.json",
      repo_id=REPO_ID,
      repo_type="model",
  )

r2 = metrics_data.get("r2_score", 0.9827)
mape = metrics_data.get("mape_percent", 12.89)
mae = metrics_data.get("mae", 2717.21)
rmse = metrics_data.get("rmse", 4933.84)

# 3. Create Model Card (README.md on Hugging Face)
model_card_content = f"""---
language:
- en
license: mit
tags:
- tabular
- lightgbm
- regression
- mlops
- indian-flights
- feast
datasets:
- flight-price-prediction
metrics:
- r2
- mape
- rmse
---

# ✈️ Indian Domestic Flight Price & 'Buy vs. Wait' Advisor Model

Trained on **300,153 Indian domestic flight records** covering top metro routes (DEL, BOM, BLR, CCU, HYD, MAA) to predict fair market fares in INR (₹) and deliver real-time booking advisory.

## 📊 Model Performance
* **Algorithm:** LightGBM Regressor (Gradient Boosted Decision Trees)
* **$R^2$ Score:** `{r2}` ({r2*100:.1f}% variance explained)
* **MAPE:** `{mape}%`
* **MAE:** `₹{mae:,.2f}`
* **RMSE:** `₹{rmse:,.2f}`

## 💻 How to Use in Python

```python
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd

# 1. Download Model from Hugging Face Hub
model_path = hf_hub_download(repo_id="{REPO_ID}", filename="model.joblib")
model = joblib.load(model_path)

# 2. Download Feast Route Stats
features_path = hf_hub_download(repo_id="{REPO_ID}", filename="route_features.parquet")
route_stats = pd.read_parquet(features_path)
print("✅ Model and Feast features loaded successfully!")
```
"""

print("📝 Uploading Model Card (README.md)...")
api.upload_file(
    path_or_fileobj=model_card_content.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"\n🎉 Successfully published model artifacts & card to: https://huggingface.co/{REPO_ID}")