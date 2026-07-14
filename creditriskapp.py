import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="💳",
    layout="wide"
)

# =====================================================
# LOAD MODEL FILES
# =====================================================
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("columns.pkl")
model_name = joblib.load("model_name.pkl")

# =====================================================
# HEADER
# =====================================================
st.title("💳 Credit Risk Prediction System")
st.caption("Machine Learning-Based Loan Default Risk Assessment")

st.markdown("---")

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("Applicant Information")

loan_amnt = st.sidebar.number_input(
    "Loan Amount (₦)",
    min_value=500.0,
    max_value=40000.0,
    value=12450.0,
)

term = st.sidebar.selectbox(
    "Loan Term",
    ["36 months", "60 months"]
)

int_rate = st.sidebar.slider(
    "Interest Rate (%)",
    5.31,
    30.99,
    12.79,
)

emp_length = st.sidebar.slider(
    "Employment Length (Years)",
    0,
    10,
    6,
)

annual_inc = st.sidebar.number_input(
    "Annual Income (₦)",
    min_value=1000.0,
    max_value=500000.0,
    value=65000.0,
)

dti = st.sidebar.slider(
    "Debt-to-Income Ratio (%)",
    0.0,
    50.0,
    17.5,
)

grade = st.sidebar.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    ["MORTGAGE", "OWN", "RENT", "OTHER", "NONE"]
)

predict = st.sidebar.button("Predict Risk")

# =====================================================
# PREDICTION
# =====================================================
if predict:

    term_value = 36 if term == "36 months" else 60

    input_data = {
        "loan_amnt": loan_amnt,
        "term": term_value,
        "int_rate": int_rate,
        "emp_length": emp_length,
        "annual_inc": annual_inc,
        "dti": dti,
    }

    # One-hot encoding
    for g in ["B", "C", "D", "E", "F", "G"]:
        input_data[f"grade_{g}"] = 1 if grade == g else 0

    for h in ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]:
        input_data[f"home_ownership_{h}"] = 1 if home_ownership == h else 0

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    probability = model.predict_proba(input_scaled)[0][1]

    # =====================================================
    # RISK CLASSIFICATION
    # =====================================================

    if probability < 0.30:

        risk = "🟢 LOW RISK"

        assessment = (
            "The applicant demonstrates a relatively low likelihood of loan "
            "default based on the financial characteristics provided. "
            "The application may proceed through the standard lending review process."
        )

        recommendation = (
            "The applicant appears creditworthy based on the model prediction "
            "and may be considered for loan approval, subject to standard lending "
            "review procedures."
        )

        st.success(risk)

    elif probability < 0.60:

        risk = "🟡 MODERATE RISK"

        assessment = (
            "The applicant presents a moderate probability of default. "
            "Additional verification of income, employment or supporting "
            "financial documentation is recommended before making a lending decision."
        )

        recommendation = (
            "The applicant presents moderate credit risk. Additional review "
            "or supporting documentation may be required before a lending "
            "decision is made."
        )

        st.warning(risk)

    else:

        risk = "🔴 HIGH RISK"

        assessment = (
            "The applicant exhibits a high predicted probability of loan default. "
            "A comprehensive credit assessment is recommended before considering "
            "loan approval."
        )

        recommendation = (
            "The applicant exhibits a high probability of default according to "
            "the trained model. Further financial assessment is recommended "
            "before approval."
        )

        st.error(risk)

    # =====================================================
    # TOP METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Decision", risk.replace("🟢 ","").replace("🟡 ","").replace("🔴 ",""))
    col2.metric("Risk Probability", f"{probability:.2%}")
    col3.metric("Model Used", model_name)
    col4.metric("Loan Grade", grade)

    st.progress(float(probability))

    st.markdown("---")

    # =====================================================
    # ASSESSMENT
    # =====================================================

    st.subheader("📊 Credit Risk Assessment")
    st.info(assessment)

    st.markdown("---")

    # =====================================================
    # APPLICANT SUMMARY
    # =====================================================

    st.subheader("👤 Applicant Summary")

    summary = pd.DataFrame({
        "Field": [
            "Loan Amount",
            "Loan Term",
            "Interest Rate",
            "Employment Length",
            "Annual Income",
            "Debt-to-Income Ratio",
            "Loan Grade",
            "Home Ownership"
        ],
        "Value": [
            f"₦{loan_amnt:,.2f}",
            term,
            f"{int_rate:.2f}%",
            f"{emp_length} years",
            f"₦{annual_inc:,.2f}",
            f"{dti:.2f}%",
            grade,
            home_ownership
        ]
    })

    st.table(summary)

    st.markdown("---")

    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.subheader("📌 Recommendation")
    st.info(recommendation)

    st.markdown("---")

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    with st.expander("ℹ Model Information"):

        st.write(f"**Algorithm:** {model_name}")
        st.write("**Prediction Method:** Probability-Based Classification")
        st.write("**Dataset Size:** Approximately 982,000 loan records")
        st.write("**Features Used:** 17")
        st.write(
            "**Risk Categories:** "
            "Low Risk (<30%), Moderate Risk (30–60%), High Risk (>60%)"
        )

st.markdown("---")
st.caption(
    "Credit Risk Prediction System | Final Year Project | "
    "Department of Computer Engineering"
)