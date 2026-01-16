import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_model.pkl")

st.title("Business Loan Approval Prediction App")
st.write("Enter Your Details below to check if your loan will be Approved")

employed = st.selectbox("Are you employed?", ["Yes", "No"])
annual_income = st.number_input("Annual Income in INR", min_value=800000)
loan_amt = st.number_input("Loan Amount in INR", min_value=2000000)
cibil = st.number_input("CIBIL Score", min_value=300, max_value=900)
assets = st.number_input("Total Assets in INR", min_value=2000000)

if st.button("Predict"):

    employed_val = 1 if employed.lower() == "yes" else 0

    if assets < loan_amt:
        st.error("Loan Rejected! (Assets lower than Loan Amount)")
    else:
        new_data = np.array([[employed_val, annual_income, loan_amt, cibil, assets]])

        result = model.predict(new_data)

        if result[0] == 1:
            st.success("Loan Approved!")
        else:
            st.error("Loan Rejected!")




