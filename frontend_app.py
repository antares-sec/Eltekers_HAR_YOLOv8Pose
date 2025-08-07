# Updated frontend_app.py
import streamlit as st
import requests
import os
import tempfile
import time

# --- Konfigurasi ---
MIDDLEWARE_API_URL = "https://9a63df1a54f6.ngrok-free.app" # <--- Sesuaikan URL ini
SUBMIT_ENDPOINT = "/submit_video/"
RESULT_ENDPOINT = "/get_result/"

# --- Tampilan Aplikasi Streamlit ---
st.set_page_config(
    page_title="Aplikasi Prediksi Aktivitas Ling Tien Kung",
    page_icon="🧘‍♂️",
    layout="centered"
)

st.title("🏃‍♂️ Prediksi Aktivitas Ling Tien Kung")
st.markdown("Unggah video aktivitas Ling Tien Kung Anda untuk mendapatkan prediksi.")
st.markdown(f"*(API Middleware asinkron di: `{MIDDLEWARE_API_URL}`)*")

uploaded_file = st.file_uploader(
    "Pilih file video (MP4, AVI, MOV)",
    type=["mp4", "avi", "mov"],
    help="Ukuran file video maksimal disarankan sekitar 20-30MB untuk performa terbaik."
)

if uploaded_file is not None:
    st.subheader("Video yang Diunggah:")
    st.video(uploaded_file, format=uploaded_file.type)

    if st.button("Mulai Prediksi"):
        # Inisialisasi status di session state
        st.session_state.processing = True
        st.session_state.job_id = None
        st.session_state.error = None
        st.session_state.output_video_path = None
        
        with st.spinner("Mengunggah file..."):
            try:
                files = {"video_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Kirim permintaan POST ke endpoint asinkron
                response = requests.post(f"{MIDDLEWARE_API_URL}{SUBMIT_ENDPOINT}", files=files, timeout=60)
                response.raise_for_status()

                job_info = response.json()
                st.session_state.job_id = job_info["job_id"]
                st.info(f"**File telah diunggah!** ID Tugas: `{st.session_state.job_id}`. Menunggu hasil prediksi...")
                st.session_state.processing = True
            
            except requests.exceptions.RequestException as e:
                st.session_state.error = f"Gagal mengunggah atau terhubung ke API middleware: {e}"
                st.session_state.processing = False
        
if 'processing' in st.session_state and st.session_state.processing:
    # Polling status di background
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    job_id = st.session_state.job_id
    start_time = time.time()
    
    while st.session_state.processing:
        time.sleep(3) # Cek status setiap 3 detik
        
        try:
            status_response = requests.get(f"{MIDDLEWARE_API_URL}{RESULT_ENDPOINT}{job_id}", timeout=10)
            status_response.raise_for_status()
            status_data = status_response.json()
            
            status = status_data.get("status")
            elapsed_time = int(time.time() - start_time)
            
            if status == "completed":
                st.session_state.processing = False
                st.session_state.output_video_content = status_response.content
                status_placeholder.success(f"Video berhasil diproses! ({elapsed_time} detik)")
                progress_bar.progress(100)
                break
            elif status == "failed":
                st.session_state.processing = False
                st.session_state.error = status_data.get("error", "Kesalahan tidak diketahui.")
                status_placeholder.error(f"Pemrosesan video gagal: {st.session_state.error}")
                progress_bar.progress(100)
                break
            else:
                # Status "pending" atau "processing"
                status_placeholder.info(f"Status: {status.capitalize()}... ({elapsed_time} detik)")
                # (Opsional) Anda bisa memperbarui progress bar berdasarkan waktu atau logika lain
                progress_bar.progress(min(elapsed_time / 120, 0.9)) # Contoh progress bar (maks 2 menit)
        
        except requests.exceptions.RequestException as e:
            st.session_state.processing = False
            st.session_state.error = f"Gagal memeriksa status pekerjaan: {e}"
            status_placeholder.error(f"Gagal memeriksa status: {e}")

if 'output_video_content' in st.session_state and st.session_state.output_video_content:
    st.subheader("Video Hasil Prediksi:")
    st.video(st.session_state.output_video_content, format="video/mp4")
    st.download_button(
        label="Unduh Video Hasil",
        data=st.session_state.output_video_content,
        file_name=f"annotated_{uploaded_file.name}",
        mime="video/mp4"
    )

if 'error' in st.session_state and st.session_state.error:
    st.error(f"Terjadi kesalahan: {st.session_state.error}")