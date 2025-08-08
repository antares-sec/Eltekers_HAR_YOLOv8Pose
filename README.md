# Proyek Ling Tien Kung Human Activity Recognition
Proyek ini adalah implementasi dari sistem pengenalan aktivitas manusia (Human Activity Recognition/HAR) untuk gerakan-gerakan Ling Tien Kung. Sistem ini menggunakan arsitektur Spatio-Temporal Graph Convolutional Network (STGCN) untuk memproses data keypoint dari video, yang diekstrak menggunakan model YOLO-Pose.

# Struktur Proyek
Struktur folder proyek ini diatur sebagai berikut:

``` markdown
├── __pycache__/
├── models/
│   ├── yolov8n-pose.pt
│   ├── stgcn_model_cpu.pth
│   └── class_mapping.pkl
├── middleware.py
├── frontend_app.py
├── requirements.txt
├── README.md
└── utils.py
```

# Penjelasan File dan Folder
**middleware.py**: Berisi implementasi utama API menggunakan framework FastAPI. API ini menerima video, memprosesnya secara asinkron menggunakan model yang telah dilatih, dan mengembalikan video yang telah di-anotasi. Proses pemodelan dilakukan di latar belakang menggunakan multiprocessing untuk menghindari blocking pada server.

**frontend_app.py**: Aplikasi web yang dibangun dengan Streamlit. File ini berfungsi sebagai antarmuka pengguna (UI) untuk berinteraksi dengan API middleware.py. Pengguna dapat mengunggah video melalui aplikasi ini dan melihat hasilnya.

**utils.py**: Berisi semua fungsi dan kelas pendukung yang diperlukan, termasuk arsitektur model STGCN, kelas InferenceDataPreprocessor untuk ekstraksi keypoint, dan fungsi untuk membangun adjacency matrix.

**models/**: Folder ini berfungsi sebagai tempat penyimpanan untuk semua model yang telah dilatih (.pt, .pth) dan file pemetaan kelas (.pkl) yang digunakan oleh middleware.py.


# Persiapan
Sebelum menjalankan proyek, pastikan Anda telah menginstal semua pustaka yang diperlukan.

## Clone file: 
```git
git clone https://github.com/antares-sec/Eltekers_HAR_YOLOv8Pose.git
```

## Instal pustaka:
``` python
pip install -r requirements.txt
```
Pastikan model tersedia: Pastikan file model yang diperlukan (seperti **yolov8n-pose.pt, stgcn_model_cpu.pth, dan class_mapping.pkl**) ada di dalam folder **models/**.

# Cara Menjalankan API (Tanpa Frontend)
Untuk menjalankan API saja, yang berguna untuk pengujian atau integrasi dengan aplikasi lain, gunakan perintah Uvicorn. Pastikan Anda berada di direktori utama proyek.
``` python
uvicorn middleware:app --reload
```
**middleware**: Nama file Python (tanpa .py).

**app**: Nama objek FastAPI yang diinisialisasi dalam file middleware.py.

**--reload**: Opsi untuk mengaktifkan auto-reload saat terjadi perubahan pada kode.

Setelah berjalan, Anda dapat mengakses dokumentasi interaktif API di http://127.0.0.1:8000/docs.

# Cara Menjalankan Aplikasi (Dengan Frontend)
Untuk menjalankan aplikasi lengkap dengan antarmuka web Streamlit, pastikan API sudah berjalan terlebih dahulu di terminal terpisah.

## Langkah 1: Jalankan API (di Terminal 1)
```python
uvicorn middleware:app --reload
```
## Langkah 2: Jalankan Frontend (di Terminal 2)
Buka terminal baru, navigasikan ke direktori proyek, dan jalankan Streamlit:
```python
streamlit run frontend_app.py
```
Aplikasi web akan terbuka di browser Anda, biasanya di **http://localhost:8501**.
