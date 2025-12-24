import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📱")

# 1. Load Model
@st.cache_resource
def load_model():
    try:
        # Pastikan file ini ada di folder yang sama
        model = joblib.load('model_churn_terbaik.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Judul
st.title("📱 Telco Customer Churn Prediction")
st.write("Isi data pelanggan di bawah ini untuk memprediksi potensi Churn.")

if model is not None:
    with st.form("churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Profil Pelanggan")
            # User input TEKS, tapi nanti kita ubah ke ANGKA
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"]) 
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (Bulan)", min_value=0, value=12)

            st.subheader("Layanan Telepon")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

        with col2:
            st.subheader("Layanan Internet")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

        st.subheader("Tagihan & Pembayaran")
        col3, col4 = st.columns(2)
        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        with col4:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        # --- PREPROCESSING MANUAL (PENTING!) ---
        # Kita harus mengubah input Teks menjadi Angka sesuai dengan LabelEncoder di Notebook Anda
        
        # 1. Mapping sederhana (Yes/No -> 1/0)
        binary_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
        
        # 2. Mapping spesifik (sesuai urutan LabelEncoder biasanya urut abjad)
        gender_map = {"Female": 0, "Male": 1}
        internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        payment_map = {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3
        }

        # Buat DataFrame dengan data yang SUDAH DI-ENCODE (Angka)
        # Urutan kolom ini HARUS SAMA PERSIS dengan X_train.columns di notebook
        input_data = pd.DataFrame({
            'gender': [gender_map[gender]],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [binary_map.get(partner, 0)],
            'Dependents': [binary_map.get(dependents, 0)],
            'tenure': [tenure],
            'PhoneService': [binary_map.get(phone_service, 0)],
            'MultipleLines': [1 if multiple_lines == "Yes" else 0], # Asumsi simple mapping, cek notebook jika pakai LabelEncoder murni (0,1,2)
            'InternetService': [internet_map[internet_service]],
            'OnlineSecurity': [1 if online_security == "Yes" else 0],
            'OnlineBackup': [1 if online_backup == "Yes" else 0],
            'DeviceProtection': [1 if device_protection == "Yes" else 0],
            'TechSupport': [1 if tech_support == "Yes" else 0],
            'StreamingTV': [1 if streaming_tv == "Yes" else 0],
            'StreamingMovies': [1 if streaming_movies == "Yes" else 0],
            'Contract': [contract_map[contract]],
            'PaperlessBilling': [binary_map.get(paperless_billing, 0)],
            'PaymentMethod': [payment_map[payment_method]],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        # --- PREDIKSI ---
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.write("---")
            if prediction == 1:
                st.error(f"⚠️ Prediksi: CHURN (Berhenti). Probabilitas: {probability:.2%}")
            else:
                st.success(f"✅ Prediksi: TIDAK CHURN (Setia). Probabilitas: {probability:.2%}")
                
        except Exception as e:
            st.warning("Terjadi kesalahan pada input data. Pastikan format model cocok.")
            st.error(f"Error detail: {e}")

else:
    st.info("Menunggu file model di-upload...")
