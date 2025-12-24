import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📱")

# 1. Load Model
@st.cache_resource
def load_model():
    try:
        # Pastikan nama file ini SAMA PERSIS dengan yang ada di GitHub
        model = joblib.load('model_churn_terbaik.pkl')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file 'model_churn_terbaik.pkl' sudah diupload ke GitHub. Error: {e}")
        return None

model = load_model()

st.title("📱 Telco Customer Churn Prediction")

if model:
    # 2. Form Input Lengkap (19 Fitur)
    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Profil Pelanggan")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (Lama Langganan - Bulan)", min_value=0, value=12)
            
            st.subheader("Layanan Telepon")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
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
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col4:
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

        submitted = st.form_submit_button("Prediksi Sekarang")

    # 3. Proses Prediksi
    if submitted:
        # DataFrame harus punya nama kolom yang SAMA PERSIS dengan notebook latihan
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.write("---")
            st.subheader("Hasil Prediksi:")
            if prediction == 1:
                st.error(f"⚠️ **POTENSI CHURN (Berhenti)**")
                st.metric("Probabilitas", f"{probability:.2%}")
            else:
                st.success(f"✅ **AMAN (Tidak Churn)**")
                st.metric("Probabilitas", f"{probability:.2%}")
        except Exception as e:
            st.error(f"Terjadi error: {e}")
else:
    st.info("Menunggu file model...")
