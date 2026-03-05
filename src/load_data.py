import pandas as pd

# Load the dataset
df = pd.read_csv("data/churn.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Show column information
print("\nDataset Info:")
print(df.info())

# Show dataset size
print("\nDataset Shape:")
print(df.shape)