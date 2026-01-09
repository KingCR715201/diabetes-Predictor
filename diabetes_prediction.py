# diabetes_prediction.py
# ----------------------------------------------------

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
# Make sure diabetes.csv is in the same directory
data = pd.read_csv("/kaggle/input/diabetes-csv/diabetes.csv")

# -------------------------------
# STEP 2: Handle Invalid Zero Values
# These columns cannot be zero medically
# -------------------------------
invalid_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace zero with NaN
data[invalid_zero_columns] = data[invalid_zero_columns].replace(0, np.nan)

# Fill NaN values using median (better than mean for insulin)
for col in invalid_zero_columns:
    data[col] = data[col].fillna(data[col].median())

# -------------------------------
# STEP 3: Split Features and Target
# -------------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# -------------------------------
# STEP 4: Train-Test Split
# Stratified to maintain class balance
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# STEP 5: Train Random Forest Model
# Random Forest handles outliers and feature importance better
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# STEP 6: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# STEP 7: Show Feature Importance
# This proves insulin now matters
# -------------------------------
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# -------------------------------
# STEP 8: Save Model
# -------------------------------
with open("diabetes_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

# -------------------------------
# STEP 9: Load Model for Prediction
# -------------------------------
with open("diabetes_rf_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# -------------------------------
# STEP 10: Test with Sample Input
# -------------------------------
sample_input = np.array([
    5,      # Pregnancies
    160,    # Glucose
    78,     # BloodPressure
    30,     # SkinThickness
    200,    # Insulin
    34.5,   # BMI
    0.9,    # DiabetesPedigreeFunction
    45      # Age
]).reshape(1, -1)

# Predict
prediction = loaded_model.predict(sample_input)[0]
probability = loaded_model.predict_proba(sample_input)[0][1]

# Output result
print("\nPrediction Result:")
if prediction == 1:
    print("Diabetic")
else:
    print("Non-Diabetic")

print("Diabetes Probability:", probability)
