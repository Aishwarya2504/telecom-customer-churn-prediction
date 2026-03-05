import pandas as pd

df = pd.read_csv("data/churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing rows
df = df.dropna()

# Remove customerID column
df = df.drop("customerID", axis=1)

# Convert target variable
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

print("Processed dataset shape:")
print(df.shape)

print("\nFirst 10 columns:")
print(df.columns[:10])