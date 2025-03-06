import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("bridge_data.csv")

df = df.drop(columns=["Location"])

from sklearn.preprocessing import LabelEncoder

# Encode target variable
df["Collapse_Status"] = df["Collapse_Status"].map({"Standing": 0, "Collapsed": 1})

# Identify categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

# Apply Label Encoding to categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for reference

# Display transformed dataset
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Drop non-essential columns
df_model = df.drop(columns=["Bridge_ID"])  # ID and Location are not useful

# Separate features and target
X = df_model.drop(columns=["Collapse_Status"])
y = df_model["Collapse_Status"]

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check the shapes
X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Tr
# Train Random Forest model
rf_model = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluation
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_report = classification_report(y_test, y_pred_rf)

rf_accuracy, rf_report

import numpy as np

# New data (provided instance)
new_data = {
    "Age (years)": 47,
    "Material": "Concrete",
    "Length (m)": 50.19501950195019,
    "Width (m)": 5.004500450045004,
    "Height (m)": 10.009000900090008,
    "Traffic_Volume (vehicles/day)": 100.99009900990099,
    "Weather_Conditions": "Sunny",
    "Water_Flow_Rate (m³/s)": 0.5000500050005,
    "Maintenance_History": "2010-01-02",
    "Stress (MPa)": 0.010001000100010001,
    "Strain (%)": 0.001000100010001,
    "Tensile_Strength (MPa)": 200.0800080008001,
    "Rainfall (mm)": 0.05000500050005,
    "Material_Composition": "Concrete 80%, Wood 20%",
    "Bridge_Design": "Beam",
    "Construction_Quality": "Good",
    "Temperature (°C)": -29.99199919991999,
    "Humidity (%)": 0.010001000100010001,
}

# Convert to DataFrame
new_df = pd.DataFrame([new_data])

# Encode categorical variables
for col in categorical_columns:
    if col in new_df.columns:
        new_df[col] = label_encoders[col].transform(new_df[col])

# Standardize numerical features
new_data_scaled = scaler.transform(new_df)

# Predict using Random Forest model
prediction = rf_model.predict(new_data_scaled)

# Convert prediction to label
result = "Collapsed" if prediction[0] == 1 else "Standing"
result

# New data instance 2
new_data_2 = {
    "Age (years)": 20,
    "Material": "Wood",
    "Length (m)": 1610.15601560156,
    "Width (m)": 41.003600360036,
    "Height (m)": 82.007200720072,
    "Traffic_Volume (vehicles/day)": 8020.79207920792,
    "Weather_Conditions": "Rainy",
    "Water_Flow_Rate (m³/s)": 4000.400040004,
    "Maintenance_History": "2031-11-27",
    "Stress (MPa)": 80.00800080008001,
    "Strain (%)": 8.000800080008,
    "Tensile_Strength (MPa)": 840.0640064006401,
    "Rainfall (mm)": 400.0400040004,
    "Material_Composition": "Concrete 80%, Wood 20%",
    "Bridge_Design": "Truss",
    "Construction_Quality": "Good",
    "Temperature (°C)": 34.006400640064,
    "Humidity (%)": 80.00800080008001,
}

# Convert to DataFrame
new_df_2 = pd.DataFrame([new_data_2])

# Encode categorical variables
for col in categorical_columns:
    if col in new_df_2.columns:
        new_df_2[col] = label_encoders[col].transform(new_df_2[col])

# Standardize numerical features
new_data_scaled_2 = scaler.transform(new_df_2)

# Predict using Random Forest model
prediction_2 = rf_model.predict(new_data_scaled_2)

# Convert prediction to label
result_2 = "Collapsed" if prediction_2[0] == 1 else "Standing"
result_2

import joblib

# Save the trained Random Forest model and scaler
model_path = "rf_bridge_model.pkl"
scaler_path = "scaler.pkl"
encoder_path = "label_encoders.pkl"

joblib.dump(rf_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoders, encoder_path)

model_path, scaler_path, encoder_path


