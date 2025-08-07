# Updated middleware.py
import os
import shutil
import uuid
import httpx
import sys
import asyncio
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask # Untuk menjalankan task di background

# Configuration
sys.path.append(".")

# Initialize API
app = FastAPI(
    title="Ling Tien Kung HAR Middleware API",
    description="Middleware API to manage asynchronous video processing",
    version="2.0.0"
)

# COLAB WORKER CONFIG
# NOTE : PLEASE CHANGE THE ADDRESS BASED ON NGROK OUTPUT ON YOUR COLAB
COLAB_WORKER_URL = "https://96aca2ef08c1.ngrok-free.app" # <-- Ganti dengan URL Colab Worker Anda
PROCESS_ENDPOINT = "/predict_activity/" # Endpoint di Colab Worker

TEMP_DIR = "temp_middleware_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# In-memory store for jobs
# Untuk aplikasi produksi, Anda akan menggunakan database atau Redis di sini.
JOB_STATUS = {}
JOB_RESULTS = {}

# Fungsi Asinkron untuk memproses video di background
async def process_video_in_background(job_id: str, input_video_path: str, video_filename: str, video_content_type: str):
    """
    Fungsi ini dijalankan di background untuk berkomunikasi dengan Colab Worker.
    """
    try:
        JOB_STATUS[job_id] = {"status": "processing"}
        print(f"[{job_id}] Mengirim video ke Colab Worker: {COLAB_WORKER_URL}{PROCESS_ENDPOINT}")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(input_video_path, "rb") as f:
                files = {'file': (video_filename, f, video_content_type)}
                response = await client.post(f"{COLAB_WORKER_URL}{PROCESS_ENDPOINT}", files=files)
        
        response.raise_for_status()
        
        # Simpan video yang diterima dari colab worker
        output_filename = f"annotated_{video_filename}"
        output_video_path = os.path.join(TEMP_DIR, f"{job_id}_{output_filename}")
        
        with open(output_video_path, "wb") as buffer:
            buffer.write(response.content)
            
        print(f"[{job_id}] Video hasil diproses dan disimpan di: {output_video_path}")
        
        # Perbarui status dan simpan path hasil
        JOB_RESULTS[job_id] = output_video_path
        JOB_STATUS[job_id] = {"status": "completed"}
        
    except httpx.HTTPStatusError as e:
        error_msg = f"[{job_id}] Error dari Colab Worker: {e.response.status_code} - {e.response.text}"
        print(error_msg)
        JOB_STATUS[job_id] = {"status": "failed", "error": error_msg}
    except Exception as e:
        error_msg = f"[{job_id}] Terjadi kesalahan tak terduga: {e}\n{traceback.format_exc()}"
        print(error_msg)
        JOB_STATUS[job_id] = {"status": "failed", "error": str(e)}
    finally:
        # Hapus file input sementara
        if os.path.exists(input_video_path):
            os.remove(input_video_path)
        print(f"[{job_id}] File input sementara telah dihapus di middleware.")

@app.post("/submit_video/", status_code=202) # Status 202 (Accepted)
async def submit_video_for_processing(video_file: UploadFile = File(...)):
    """
    Menerima video dan memulai proses di background. Mengembalikan Job ID segera.
    """
    if not video_file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Hanya format video MP4, AVI, atau MOV yang diterima.")
    
    job_id = str(uuid.uuid4())
    input_video_path = os.path.join(TEMP_DIR, f"{job_id}_{video_file.filename}")

    try:
        # Simpan video input secara lokal
        with open(input_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        # Jalankan proses video ke Colab Worker di background
        # BackgroundTask adalah cara FastAPI untuk menjalankan fungsi asinkron tanpa menunggu.
        asyncio.create_task(
            process_video_in_background(job_id, input_video_path, video_file.filename, video_file.content_type)
        )
        
        return JSONResponse(
            content={"job_id": job_id, "message": "File telah diunggah. Video sedang diproses di background."},
            status_code=202
        )
        
    except Exception as e:
        print(f"Terjadi kesalahan saat mengunggah video: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Gagal memulai proses video.")

@app.get("/get_result/{job_id}")
async def get_processing_result(job_id: str):
    """
    Memeriksa status pekerjaan dan mengembalikan hasilnya jika sudah selesai.
    """
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job ID tidak ditemukan.")

    status_data = JOB_STATUS[job_id]
    status = status_data.get("status")

    if status == "completed":
        output_video_path = JOB_RESULTS.get(job_id)
        if not output_video_path or not os.path.exists(output_video_path):
            return {"status": "failed", "error": "File hasil tidak ditemukan."}
        
        # Hapus data dari in-memory store setelah dikirim
        del JOB_STATUS[job_id]
        del JOB_RESULTS[job_id]
        
        # Kirim file video
        return FileResponse(
            path=output_video_path,
            media_type="video/mp4",
            filename=os.path.basename(output_video_path),
            background=BackgroundTask(lambda: os.remove(output_video_path)) # Hapus file setelah dikirim
        )
    else:
        return {"status": status, "error": status_data.get("error")}