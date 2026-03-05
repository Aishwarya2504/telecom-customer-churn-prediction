import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/churn.csv")

# Preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop("customerID", axis=1)

df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_

# Create dataframe
features = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

features = features.sort_values(by="Importance", ascending=False)

print(features.head(10))

# Plot
plt.figure(figsize=(10,6))
plt.barh(features["Feature"][:10], features["Importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Important Features for Churn Prediction")
plt.show()