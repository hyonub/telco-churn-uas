import streamlit as st
import pandas as pd
import joblib

# Load Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('model_churn_terbaik.pkl')
    except:
        return None

model = load_model()

st.title("Telco Churn Prediction")

if model:
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col2:
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            monthly_charges = st.number_input("Monthly Charges", value=50.0)
            total_charges = st.number_input("Total Charges", value=500.0)
            # Tambahkan fitur dummy untuk fitur lain yang tidak diinput manual agar sesuai model
            # (Ini penyederhanaan agar aplikasi jalan)

        submit = st.form_submit_button("Predict")

    if submit:
        # Siapkan data (pastikan urutan kolom sama persis dengan waktu training)
        # Karena kode ini manual, pastikan input sesuai dengan fitur model Anda.
        st.info("Prediksi sedang diproses...")
        # Note: Agar akurat 100%, pastikan semua kolom input ada.
else:
    st.error("File 'model_churn_terbaik.pkl' tidak ditemukan. Upload file model ke GitHub.")