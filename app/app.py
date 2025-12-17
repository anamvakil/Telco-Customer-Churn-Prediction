import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

# --------------------------------------------------
# 1) Load data (from repo)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df

df = load_data()

# --------------------------------------------------
# 2) Features used for model + UI
# --------------------------------------------------
features = ["Contract", "PaymentMethod", "InternetService", "tenure", "MonthlyCharges"]
X = df[features]
y = df["Churn"]

numeric_features = ["tenure", "MonthlyCharges"]
categorical_features = ["Contract", "PaymentMethod", "InternetService"]

# --------------------------------------------------
# 3) Build preprocessing + model (cached)
# --------------------------------------------------
@st.cache_resource
def train_model(X, y):
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
    return model, test_accuracy

model, test_accuracy = train_model(X, y)

# --------------------------------------------------
# 4) Streamlit UI
# --------------------------------------------------
st.title("📉 Telco Customer Churn Predictor")
st.write(
    "This demo app uses a **Logistic Regression** model trained on the Telco Customer Churn dataset "
    "to estimate the probability that a customer will churn."
)

st.metric("Model Test Accuracy", f"{test_accuracy:.2%}")

st.subheader("Input Customer Details")

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

tenure = st.slider(
    "Tenure (months)",
    min_value=int(df["tenure"].min()),
    max_value=int(df["tenure"].max()),
    value=12,
    step=1,
)

monthly_charges = st.slider(
    "Monthly Charges ($)",
    min_value=float(df["MonthlyCharges"].min()),
    max_value=float(df["MonthlyCharges"].max()),
    value=70.0,
    step=1.0,
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

# --------------------------------------------------
# 5) Risk band helper
# --------------------------------------------------
def risk_band(message, color):
    st.markdown(
        f"""
        <div style="
            padding: 14px;
            border-radius: 14px;
            background: {color};
            color: white;
            font-weight: 800;
            text-align: center;
            font-size: 1.05rem;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# 6) Prediction (ONLY when button clicked)
# --------------------------------------------------
st.subheader("Prediction")

if st.button("Predict Churn Risk"):
    proba = float(model.predict_proba(input_data)[0][1])  # probability of churn (1)

    st.write(f"Churn Probability: **{proba:.2%}**")

    # ✅ Your exact thresholds + categories
    if proba >= 0.55:
        risk_band("High churn risk", "#d32f2f")       # red
    elif proba >= 0.35:
        risk_band("Moderate churn risk", "#f57c00")   # orange
    else:
        risk_band("Low churn risk", "#388e3c")        # green
else:
    st.info("Adjust the inputs, then click **Predict Churn Risk**.")
