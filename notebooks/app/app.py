

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. Load and clean original Telco dataset
# --------------------------------------------------
df = pd.read_csv("D:/Chrome Downloads/Business Intelligence/Project/archive/Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric and drop rows with missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# Convert Churn to 0/1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Features used for the model / UI
features = ["Contract", "PaymentMethod", "InternetService", "tenure", "MonthlyCharges"]
X = df[features]
y = df["Churn"]

numeric_features = ["tenure", "MonthlyCharges"]
categorical_features = ["Contract", "PaymentMethod", "InternetService"]

# --------------------------------------------------
# 2. Build preprocessing + model
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

# --------------------------------------------------
# 3. Streamlit UI
# --------------------------------------------------
st.title("Telco Customer Churn Predictor")
st.write(
    "This demo app uses a Logistic Regression model trained on the "
    "Telco Customer Churn dataset to predict whether a customer is likely to churn."
)

st.write(f"Model Test Accuracy: {test_accuracy:.2%}")

st.subheader("Input Customer Details")

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tenure = st.slider(
    "Tenure (months)",
    min_value=int(df["tenure"].min()),
    max_value=int(df["tenure"].max()),
    value=12,
    step=1
)

monthly_charges = st.slider(
    "Monthly Charges ($)",
    min_value=float(df["MonthlyCharges"].min()),
    max_value=float(df["MonthlyCharges"].max()),
    value=70.0,
    step=1.0
)

# One-row DataFrame for prediction
input_data = pd.DataFrame(
    {
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "InternetService": [internet_service],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
    }
)

if st.button("Predict Churn"):
    proba = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.write(f"Prediction: {'Will Churn' if pred == 1 else 'Will Not Churn'}")
    st.write(f"Churn Probability: {proba * 100:.2f}%")

    if proba >= 0.7:
        st.warning("High churn risk.")
    elif proba >= 0.4:
        st.info("Moderate churn risk.")
    else:
        st.success("Low churn risk.")





