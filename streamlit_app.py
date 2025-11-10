import streamlit as st
import librosa
import tsfel
import pandas as pd
import numpy as np
import joblib
import warnings
import tempfile
import subprocess
import os
import io
from audiorecorder import audiorecorder  # Ini sudah benar

# Mengabaikan peringatan dari librosa (opsional)
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# --- 1. KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Prediksi Suara Fikri & Fauzan",
    page_icon="🎙️"
)

TARGET_SR = 16000 # Definisikan sample rate target secara global

# --- 2. MEMUAT "OTAK" MODEL (HANYA SEKALI) ---
@st.cache_resource
def load_assets():
    """Memuat semua aset model yang diperlukan."""
    print("Memuat aset model...") 
    try:
        model = joblib.load("model_suara.joblib")
        imputer = joblib.load("imputer.joblib")
        selected_features_names = joblib.load("selected_features.joblib")
        tsfel_cfg = tsfel.get_features_by_domain()
        
        if not isinstance(selected_features_names, list):
            st.error("Error: 'selected_features.joblib' bukan sebuah list. Pastikan file sudah benar.")
            return None, None, None, None
            
        return model, imputer, selected_features_names, tsfel_cfg
    except FileNotFoundError as e:
        st.error(f"ERROR: File model tidak ditemukan. Pastikan file .joblib ada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

# Panggil fungsi untuk memuat aset
model, imputer, selected_features_names, tsfel_cfg = load_assets()

# --- 3. FUNGSI INTI PREDIKSI ---
def run_prediction_pipeline(signal, sr):
    """
    Menjalankan pipeline TSFEL, imputasi, dan prediksi pada signal audio.
    """
    try:
        # 1. Ekstraksi Fitur (TSFEL)
        features = tsfel.time_series_features_extractor(tsfel_cfg, signal, fs=sr, verbose=0)
        
        # 2. Seleksi Fitur
        features_reindexed = features.reindex(columns=selected_features_names)
        
        # 3. Imputasi (Membersihkan data)
        features_no_inf = features_reindexed.replace([np.inf, -np.inf], np.nan)
        features_imputed = imputer.transform(features_no_inf)
        
        # 4. Prediksi
        prediction = model.predict(features_imputed)
        pred_array = prediction[0]
        
        # 5. Interpretasi Hasil
        kata = "Buka" if pred_array[0] == 1 else "Tutup"
        pembicara = "Fikri" if pred_array[2] == 1 else "Fauzan"
        
        return kata, pembicara
        
    except Exception as e:
        st.error(f"Terjadi error saat menjalankan pipeline prediksi: {e}")
        return None, None

# --- 4. FUNGSI UNTUK MEMPROSES FILE UPLOAD (DENGAN FFMPEG) ---
def process_uploaded_file(audio_file):
    """
    Menangani file yang di-upload, mengonversi dengan FFmpeg,
    dan memanggil pipeline inti.
    """
    file_extension = os.path.splitext(audio_file.name)[-1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as input_temp:
        input_temp.write(audio_file.getvalue())
        input_temp_path = input_temp.name

    output_temp_path = input_temp_path + ".wav"

    try:
        # 1. Jalankan FFmpeg untuk konversi
        command = [
            "ffmpeg", "-i", input_temp_path,
            "-ac", "1", "-ar", str(TARGET_SR),
            output_temp_path, "-y"
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 2. Load file .wav yang sudah dikonversi
        signal, sr = librosa.load(output_temp_path, sr=TARGET_SR, mono=True)
        
        # 3. Panggil pipeline inti
        return run_prediction_pipeline(signal, sr)

    except subprocess.CalledProcessError as e:
        st.error(f"Error saat konversi FFmpeg: {e.stderr.decode()}")
        return None, None
    except FileNotFoundError:
        st.error("ERROR: FFmpeg tidak ditemukan. Pastikan FFmpeg sudah terinstal.")
        return None, None
    except Exception as e:
        st.error(f"Terjadi error saat memproses file upload: {e}")
        return None, None
    finally:
        # Bersihkan file temporary
        if 'input_temp_path' in locals() and os.path.exists(input_temp_path):
            os.remove(input_temp_path)
        if 'output_temp_path' in locals() and os.path.exists(output_temp_path):
            os.remove(output_temp_path)

# --- 5. TAMPILAN ANTARMUKA (UI) STREAMLIT ---
st.title("🎙️ Aplikasi Prediksi Suara")
st.write("Prediksi suara 'buka' / 'tutup' oleh Fikri / Fauzan.")
st.write("---")

# Buat dua tab: satu untuk upload, satu untuk rekam
tab1, tab2 = st.tabs(["📤 Upload File", "🔴 Rekam Langsung"])

# --- Tab 1: Upload File (Tidak Berubah) ---
with tab1:
    st.header("Opsi 1: Upload File Audio")
    uploaded_file = st.file_uploader("Pilih file audio...", type=['wav', 'mp3', 'aac'], label_visibility="collapsed")

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("🚀 Prediksi File Upload!"):
            with st.spinner("Mengonversi dan menganalisis audio..."):
                kata, pembicara = process_uploaded_file(uploaded_file)
                
                if kata and pembicara:
                    st.success("Prediksi Berhasil!")
                    col1, col2 = st.columns(2)
                    col1.metric("Kata yang Diucapkan", kata.upper())
                    col2.metric("Pembicara", pembicara.upper())

# --- Tab 2: Rekam Langsung (MODIFIKASI BESAR DI SINI) ---
with tab2:
    st.header("Opsi 2: Rekam Suara Langsung")
    st.info("Klik tombol di bawah untuk mulai merekam. Klik lagi untuk berhenti.")

    # 1. Panggil perekam. Ini mengembalikan objek pydub.AudioSegment
    pydub_audio = audiorecorder()

    if pydub_audio is not None:
        
        # --- PERBAIKAN DI SINI ---
        # 2. Konversi objek pydub.AudioSegment ke byte WAV
        # Ini PENTING agar st.audio dan librosa bisa membacanya
        audio_bytes_io = io.BytesIO()
        pydub_audio.export(audio_bytes_io, format="wav")
        audio_bytes_io.seek(0)
        wav_audio_bytes = audio_bytes_io.getvalue()
        
        # 3. Tampilkan audio player menggunakan byte
        st.audio(wav_audio_bytes, format='audio/wav')
        
        # Tombol untuk memprediksi rekaman
        if st.button("🚀 Prediksi Rekaman!"):
            with st.spinner("Menganalisis rekaman..."):
                try:
                    # 4. Load audio dari byte yang sudah dikonversi
                    # (Kita tidak perlu lagi audio_segment, langsung pakai byte)
                    signal, sr = librosa.load(io.BytesIO(wav_audio_bytes), sr=TARGET_SR, mono=True)
                    
                    # 5. Panggil pipeline inti
                    kata, pembicara = run_prediction_pipeline(signal, sr)
                    
                    if kata and pembicara:
                        st.success("Prediksi Berhasil!")
                        col1, col2 = st.columns(2)
                        col1.metric("Kata yang Diucapkan", kata.upper())
                        col2.metric("Pembicara", pembicara.upper())
                        
                except Exception as e:
                    st.error(f"Error saat memproses rekaman: {e}")