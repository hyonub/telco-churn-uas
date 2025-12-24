import streamlit as st
import pandas as pd
import joblib

# 1. Load Model Terbaik [cite: 93]
# Pastikan file .pkl berada satu folder dengan app.py saat dideploy
model = joblib.load('model_churn_terbaik.pkl')

# Judul Aplikasi
st.title("Telco Customer Churn Prediction")
st.write("Aplikasi ini memprediksi apakah pelanggan berpotensi berhenti berlangganan (Churn) berdasarkan data profil mereka.")

# 2. Form Input Fitur [cite: 94]
# Kita bagi menjadi dua kolom agar tampilan lebih rapi
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Demografis")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen (Lansia)", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    partner = st.selectbox("Memiliki Pasangan (Partner)", ["Yes", "No"])
    dependents = st.selectbox("Memiliki Tanggungan (Dependents)", ["Yes", "No"])

    st.subheader("Informasi Layanan")
    tenure = st.number_input("Lama Berlangganan (Bulan)", min_value=0, value=12)
    phone_service = st.selectbox("Layanan Telepon", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Jenis Internet", ["DSL", "Fiber optic", "No"])

with col2:
    st.subheader("Fitur Tambahan")
    online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    st.subheader("Pembayaran")
    contract = st.selectbox("Kontrak", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Tagihan Tanpa Kertas", ["Yes", "No"])
    payment_method = st.selectbox("Metode Pembayaran", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Biaya Bulanan (Monthly Charges)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Biaya (Total Charges)", min_value=0.0, value=500.0)

# 3. Proses Prediksi [cite: 95]
if st.button("Prediksi Churn"):
    # Menyusun data input ke dalam DataFrame sesuai urutan training
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
        # Melakukan prediksi
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # 4. Tampilan Hasil Prediksi [cite: 96]
        st.write("---")
        if prediction == 1:
            st.error(f"⚠️ **Hasil: CHURN (Berhenti Berlangganan)**")
            st.write(f"Probabilitas pelanggan ini akan pergi: **{probability:.2%}**")
            st.write("Saran: Tawarkan diskon atau promosi khusus untuk mempertahankan pelanggan ini.")
        else:
            st.success(f"✅ **Hasil: TIDAK CHURN (Tetap Berlangganan)**")
            st.write(f"Probabilitas pelanggan ini akan pergi: **{probability:.2%}**")
            st.write("Pelanggan ini tampaknya setia.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
