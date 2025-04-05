# train.py

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

# Load California Housing dataset
data = fetch_california_housing(as_frame=True)
X = data.frame
y = data.target

# Rename some columns to match your API input structure
X = X.rename(columns={
    "AveRooms": "RM",
    "AveOccup": "CHAS",
    "HouseAge": "PTRATIO",
    "AveBedrms": "LSTAT",
    "Population": "B",
    "MedInc": "TAX"  # just for placeholder mapping
})

# Ensure only required features are used
features = ["CHAS", "RM", "TAX", "PTRATIO", "B", "LSTAT"]
X = X[features]

# Split, scale, and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = GradientBoostingRegressor()
model.fit(X_train_scaled, y_train)

# Save model and scaler
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "boston_housing_prediction.joblib")
joblib.dump(scaler, "outputs/scaler.joblib")

print("✅ Model and scaler saved (California Housing version)")
