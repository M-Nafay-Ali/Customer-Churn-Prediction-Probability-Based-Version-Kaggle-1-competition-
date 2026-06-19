import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model pipeline
model = joblib.load('best_churn_model.pkl')

# 2. Set up the web app title
st.title("🔮 Customer Churn Prediction Dashboard")
st.write("Enter the customer's details below to predict their probability of leaving.")

st.divider()

# 3. Create input elements matching your dataset features
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen?", [0, 1])
    Partner = st.selectbox("Has Partner?", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service?", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])

with col3:
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    # Add filler defaults for other features your pipeline expects
    OnlineSecurity = "No"
    OnlineBackup = "No"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "No"
    StreamingMovies = "No"

# 4. Trigger Prediction when button is clicked
if st.button("Predict Churn Probability", type="primary"):
    
    # Put inputs into a dictionary format matching the exact columns of X
    input_data = {
        'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner], 
        'Dependents': [Dependents], 'tenure': [tenure], 'PhoneService': [PhoneService], 
        'MultipleLines': [MultipleLines], 'InternetService': [InternetService], 
        'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup], 
        'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport], 
        'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies], 
        'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling], 
        'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [MonthlyCharges], 
        'TotalCharges': [TotalCharges]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Get probability output
    probability = model.predict_proba(input_df)[0][1]
    
    st.divider()
    st.subheader("Prediction Result:")
    st.metric(label="Churn Probability", value=f"{probability * 100:.2f}%")
    
    if probability > 0.5:
        st.error("⚠️ High Risk: This customer is highly likely to leave!")
    else:
        st.success("✅ Low Risk: This customer looks stable and loyal.")
