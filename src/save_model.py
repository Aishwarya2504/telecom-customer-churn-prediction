import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/churn.csv")

# Preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop("customerID", axis=1)

df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "models/churn_model.pkl")

# Save feature columns
joblib.dump(X.columns, "models/model_columns.pkl")

print("Model saved successfully!")