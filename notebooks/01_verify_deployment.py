# Databricks notebook source
# MAGIC %md
# MAGIC # CoMarketer — Verify Deployment
# MAGIC Run this notebook after `databricks bundle deploy` to verify the app is working.

# COMMAND ----------

import requests
import json

# Update this with your actual app URL after deployment
APP_URL = "https://<your-workspace>.cloud.databricks.com/apps/comarketer"

# COMMAND ----------

# Test /invocations endpoint
payload = {
    "input": [{"role": "user", "content": "Hello!"}],
    "custom_inputs": {"sp_id": "sp-igp-001"}
}

response = requests.post(
    f"{APP_URL}/invocations",
    json=payload,
    headers={"Content-Type": "application/json"}
)

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# COMMAND ----------

# Verify response contains expected client info
data = response.json()
assert "IGP" in str(data), "Expected IGP in response"
print("✅ Deployment verified — agent responds with correct client identity.")
