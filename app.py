import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model and Preprocessing Files
# -------------------------------
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Loan Approval Prediction System")
st.write("Predict whether a loan application is likely to be approved using Machine Learning.")

st.sidebar.header("About")
st.sidebar.write("""
This project uses **XGBoost** to predict loan approval.

Developed using:
- Python
- Scikit-Learn
- XGBoost
- Streamlit
""")

# -------------------------------
# User Inputs
# -------------------------------
st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", 18, 100, 25)

    person_gender = st.selectbox(
        "Gender",
        ["female", "male"]
    )

    person_education = st.selectbox(
        "Education",
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0
    )

    person_emp_exp = st.number_input(
        "Employment Experience (Years)",
        min_value=0,
        value=2
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

with col2:

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=10.0
    )

    loan_percent_income = st.number_input(
        "Loan Percent Income",
        min_value=0.0,
        value=0.20
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length",
        min_value=0,
        value=5
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700
    )

    previous_loan_defaults_on_file = st.selectbox(
        "Previous Loan Default",
        ["No", "Yes"]
    )

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Loan Status"):

    user_data = pd.DataFrame({
        "person_age":[person_age],
        "person_gender":[person_gender],
        "person_education":[person_education],
        "person_income":[person_income],
        "person_emp_exp":[person_emp_exp],
        "person_home_ownership":[person_home_ownership],
        "loan_amnt":[loan_amnt],
        "loan_intent":[loan_intent],
        "loan_int_rate":[loan_int_rate],
        "loan_percent_income":[loan_percent_income],
        "cb_person_cred_hist_length":[cb_person_cred_hist_length],
        "credit_score":[credit_score],
        "previous_loan_defaults_on_file":[previous_loan_defaults_on_file]
    })

    # One-Hot Encoding
    user_data = pd.get_dummies(user_data)

    # Match training columns
    user_data = user_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scaling
    user_scaled = scaler.transform(user_data)

    # Prediction
    prediction = model.predict(user_scaled)
    probability = model.predict_proba(user_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write("### Prediction Probability")

    st.progress(float(probability[0][1]))

    st.write(f"Approval Probability : **{probability[0][1]*100:.2f}%**")
    st.write(f"Rejection Probability : **{probability[0][0]*100:.2f}%**")

st.markdown("---")
st.caption("Developed by Om Soni")