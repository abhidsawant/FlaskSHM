import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("bridge_data.csv")

# Store original column names before any modifications for future reference
original_column_names = df.drop(columns=["Collapse_Status"]).columns.tolist()

# Drop non-useful columns
df = df.drop(columns=["Location", "Bridge_ID"])  # Removing unnecessary identifiers

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

# Separate features and target
X = df.drop(columns=["Collapse_Status"])
y = df["Collapse_Status"]

# Feature Selection using Random Forest Importance
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X, y)
feature_importances = pd.Series(rf_temp.feature_importances_, index=X.columns)
selected_features = feature_importances[feature_importances > 0.01].index.tolist()  # Keep important features
X = X[selected_features]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best model from tuning
best_rf_model = grid_search.best_estimator_

# Evaluate Model
y_pred = best_rf_model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model, scaler, and encoders
joblib.dump(best_rf_model, "rf_bridge_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(selected_features, "selected_features.pkl")
# Also save original column names for reference
joblib.dump(original_column_names, "original_column_names.pkl")

# Function to make predictions
def predict_bridge_status(input_data):
    # Create a DataFrame with the input data
    new_df = pd.DataFrame([input_data])
    
    # Rename columns to match training data format if needed
    # This step ensures column names match exactly what the model was trained on
    # You might need to adjust this based on your actual column names in the dataset
    column_mapping = {}
    # Load original column names if available
    try:
        original_cols = joblib.load("original_column_names.pkl")
        # Create mapping between input keys and original column names
        # This is a simplified example - you may need to create a more specific mapping
        for input_key in input_data.keys():
            for orig_col in original_cols:
                # Find the closest match between input keys and original columns
                if input_key.lower().replace(" ", "_") in orig_col.lower().replace(" ", "_") or \
                   orig_col.lower().replace(" ", "_") in input_key.lower().replace(" ", "_"):
                    column_mapping[input_key] = orig_col
                    break
        
        if column_mapping:
            new_df = new_df.rename(columns=column_mapping)
    except:
        print("Warning: Could not load original column names. Proceeding with provided names.")
    
    # Load selected features
    selected_features = joblib.load("selected_features.pkl")
    
    # Encode categorical variables
    label_encoders = joblib.load("label_encoders.pkl")
    for col in label_encoders.keys():
        if col in new_df.columns:
            try:
                new_df[col] = label_encoders[col].transform(new_df[col])
            except ValueError as e:
                print(f"Warning: {e} for column {col}. Setting to default value -1.")
                new_df[col] = -1  # Handle unseen labels
    
    # Keep only required features
    missing_cols = set(selected_features) - set(new_df.columns)
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}. Setting to default value 0.")
        for col in missing_cols:
            new_df[col] = 0  # Default value for missing columns
    
    new_df = new_df[selected_features]  # Reorder columns to match training data
    
    # Load scaler
    scaler = joblib.load("scaler.pkl")
    
    # Standardize
    new_data_scaled = scaler.transform(new_df)
    
    # Load model
    model = joblib.load("rf_bridge_model.pkl")
    
    # Predict
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)
    
    result = {
        "prediction": "Collapsed" if prediction[0] == 1 else "Standing",
        "probability": probability[0][1]  # Probability of collapse
    }
    
    return result

# Example of how to use the prediction function
# Make sure these keys match exactly with your training data column names
example_input = {
    "Age": 50,  # Instead of "Age (years)"
    "Material": "Concrete",
    "Length": 120,  # Instead of "Length (m)"
    "Width": 20,
    "Height": 15,
    "Traffic_Volume": 5000,
    "Weather_Conditions": "Rainy",
    "Water_Flow_Rate": 200,
    "Maintenance_History": "Poor",  # If this was categorical in your training
    "Stress": 10,
    "Strain": 0.5,
    "Tensile_Strength": 300,
    "Rainfall": 100,
    "Material_Composition": "Concrete",  # Match exact categories used in training
    "Bridge_Design": "Arch",
    "Construction_Quality": "Average",
    "Temperature": 25,
    "Humidity": 60,
}

# For debugging purposes, print column names
print("Selected Features (required by model):", selected_features)
print("Input Features (provided for prediction):", list(example_input.keys()))

# Make prediction
result = predict_bridge_status(example_input)
print("Prediction:", result["prediction"])
print("Collapse Probability:", result["probability"])