import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# Dark theme styling
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📡 Telecom Customer Churn Prediction")

# Banner image
st.image(
    "https://images.unsplash.com/photo-1556745757-8d76bdb6984b",
    width=600
)

st.markdown("---")

# Load dataset (for feature importance only)
df = pd.read_csv("data/churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.drop("customerID", axis=1)

df["Churn"] = df["Churn"].map({"Yes":1, "No":0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# LOAD SAVED MODEL
try:
    model = joblib.load("models/churn_model.pkl")
    X_columns = joblib.load("models/model_columns.pkl")
except:
    # fallback if model not saved yet
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    X_columns = X.columns

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Layout columns
col1, col2 = st.columns(2)

# Customer input
with col1:
    st.subheader("Customer Details")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 20.0, 120.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 9000.0, 1000.0)
    senior = st.selectbox("Senior Citizen", [0,1])

# Prediction section
with col2:
    st.subheader("Prediction Result")

    input_data = pd.DataFrame({
        "SeniorCitizen":[senior],
        "tenure":[tenure],
        "MonthlyCharges":[monthly_charges],
        "TotalCharges":[total_charges]
    })

    # Align columns with model
    for col in X_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X_columns]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    prob = round(probability[0][1]*100,2)

    if prediction[0] == 1:
        st.error("⚠️ High churn risk")
    else:
        st.success("✅ Customer likely to stay")

    st.metric("Churn Probability", f"{prob}%")
    st.progress(prob/100)

st.markdown("---")

# Feature importance chart
st.subheader("Key Factors Influencing Churn")
st.bar_chart(importance.head(10).set_index("Feature"))

st.markdown("---")

# About section
st.subheader("About This Project")

st.write("""
This dashboard predicts whether a telecom customer is likely to discontinue their service based on their account characteristics.

The prediction model uses a **Random Forest machine learning algorithm** trained on telecom customer data. The model analyzes patterns in customer tenure, billing behavior, internet services, and payment methods.

Analysis of the dataset shows that churn is strongly associated with shorter customer tenure, higher monthly charges, fiber optic internet services, and certain payment methods.

This predictive system can help telecom providers identify customers at risk and implement proactive retention strategies.
""")