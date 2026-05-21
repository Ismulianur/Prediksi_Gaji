import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page title
st.set_page_config(page_title="Prediksi Gaji Pertama", layout="centered")

# Load assets
@st.cache_resource
def load_assets():
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return encoders, scaler, model

try:
    loaded_encoders, loaded_scaler, loaded_model = load_assets()

    st.title("🚀 Aplikasi Prediksi Gaji Pertama")
    st.write("Masukkan data di bawah ini untuk memprediksi gaji pertama Anda (dalam Juta).")

    # Form Input
    col1, col2 = st.columns(2)

    with col1:
        jenis_kelamin = st.selectbox("Jenis Kelamin", options=['L', 'P'])
        usia = st.number_input("Usia", min_value=17, max_value=60, value=25)
        pendidikan = st.selectbox("Pendidikan", options=['SMA', 'SMK', 'D3', 'S1'])
        jurusan = st.selectbox("Jurusan", options=['administrasi', 'desain grafis', 'otomotif', 'teknik las', 'teknik listrik'])

    with col2:
        durasi_jam = st.number_input("Durasi Jam Pelatihan", min_value=1.0, max_value=200.0, value=60.0)
        nilai_ujian = st.number_input("Nilai Ujian", min_value=0.0, max_value=100.0, value=85.0)
        status_bekerja = st.selectbox("Status Bekerja", options=['Belum Bekerja', 'Sudah Bekerja'])

    if st.button("Prediksi Gaji"):
        # Prepare input as DataFrame
        input_data = pd.DataFrame({
            'Jenis_Kelamin': [jenis_kelamin],
            'Usia': [float(usia)],
            'Pendidikan': [pendidikan],
            'Jurusan': [jurusan],
            'Durasi_Jam': [float(durasi_jam)],
            'Nilai_Ujian': [float(nilai_ujian)],
            'Status_Bekerja': [status_bekerja]
        })

        # Preprocessing: Encoding
        for col, le in loaded_encoders.items():
            # Apply same logic as training (strip and transform)
            input_data[col] = le.transform(input_data[col].str.strip())

        # Preprocessing: Scaling
        input_scaled = loaded_scaler.transform(input_data)

        # Prediction
        prediction = loaded_model.predict(input_scaled)

        st.success(f"### Hasil Prediksi: Rp {prediction[0]:.2f} Juta")

except FileNotFoundError:
    st.error("File pickle (model/scaler/encoder) tidak ditemukan. Pastikan file sudah diexport sebelumnya.")
