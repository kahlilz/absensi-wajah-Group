# Import library yang dibutuhkan (Flask, OpenCV, dll.)
from flask import Flask, request, jsonify, render_template_string
import os
import cv2
import sys
import numpy as np
from PIL import Image
from datetime import datetime
import pickle
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
import base64 # Untuk mengirim gambar ke template
from io import BytesIO

# --- Fungsi Path (SAMA SEPERTI APLIKASI DESKTOP) ---
def get_app_path():
    if getattr(sys, 'frozen', False): return os.path.dirname(sys.executable)
    else: return os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except Exception: base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

APP_PATH = get_app_path()
embedding_db_path = os.path.join(APP_PATH, 'database', 'embeddings.pkl')
path_ke_model_str = resource_path('yolov8n-face-lindevs.pt')

# --- INISIALISASI (Perlu dimuat sekali saja, BUKAN per request) ---
print("Memuat model YOLO...")
face_detector_model = YOLO(path_ke_model_str)
print("Memuat database embedding...")
embedding_db = {}
if os.path.exists(embedding_db_path):
    try:
        with open(embedding_db_path, 'rb') as f: embedding_db = pickle.load(f)
    except Exception as e: print(f"Gagal memuat DB: {e}")
print("Model dan DB siap.")

# --- Fungsi Pengenalan Wajah (Mirip, tapi disederhanakan) ---
def _recognize_one_face_web(face_crop, db):
    if face_crop.size == 0: return "Crop Gagal", float('inf')
    try:
        live_embedding_obj = DeepFace.represent(face_crop, model_name='SFace', enforce_detection=False, detector_backend='skip')
        live_embedding = live_embedding_obj[0]['embedding']
        if not db: return "DB Kosong", float('inf')
        min_distance = float("inf")
        recognized_id = "Tidak Dikenal"
        for user_id, data in db.items():
            if 'embeddings' in data and data['embeddings']:
                for db_embedding in data['embeddings']:
                    distance = cosine(live_embedding, db_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        recognized_id = user_id
        return recognized_id, min_distance
    except Exception as e:
        print(f"Error _recognize_one_face_web: {e}")
        return "Error Proses", float('inf')

# --- Setup Aplikasi Flask ---
app = Flask(__name__)
# Folder untuk menyimpan upload sementara
UPLOAD_FOLDER = os.path.join(APP_PATH, 'uploads_web')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Halaman Utama (Frontend HTML Sederhana) ---
# Ini hanya contoh, idealnya file HTML terpisah
html_template = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Absen Wajah Web (Demo)</title>
    <style> body { font-family: sans-serif; padding: 20px; } </style>
</head>
<body>
    <h1>Demo Absensi Wajah via Web</h1>
    <form method="post" enctype="multipart/form-data">
        <label for="file">Pilih Foto Kelas:</label><br>
        <input type="file" id="file" name="file" accept="image/*" required><br><br>
        <button type="submit">Unggah & Proses</button>
    </form>
    <hr>
    <h2>Hasil Proses:</h2>
    {% if results %}
        <p>Gambar Asli:</p>
        <img src="data:image/jpeg;base64,{{ original_image_b64 }}" alt="Original" style="max-width: 400px; height: auto;">
        <p>Wajah Terdeteksi:</p>
        <ul>
        {% for result in results %}
            <li>
                <img src="data:image/jpeg;base64,{{ result.face_image_b64 }}" alt="Face" style="width: 100px; height: auto; vertical-align: middle;">
                Status: {{ result.status }} (Keyakinan: {{ result.confidence }}%) - Nama: {{ result.name }}
            </li>
        {% endfor %}
        </ul>
    {% elif message %}
        <p style="color: red;">{{ message }}</p>
    {% endif %}
</body>
</html>
"""

# --- Endpoint/Route Flask ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(html_template, message="Tidak ada file dipilih!")
        file = request.files['file']
        if file.filename == '':
            return render_template_string(html_template, message="Nama file kosong!")
        
        if file:
            # Simpan file sementara
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Baca gambar
            frame = cv2.imread(filepath)
            if frame is None:
                os.remove(filepath) # Hapus file jika gagal dibaca
                return render_template_string(html_template, message="Gagal membaca file gambar!")

            # --- [TITIK RAWAN PERFORMA 1] Deteksi Wajah ---
            results_detection = face_detector_model(frame, verbose=False, conf=0.4)
            all_boxes = results_detection[0].boxes.xyxy.cpu().numpy().astype(int)
            
            processed_results = []
            THRESHOLD_YAKIN_WEB = 0.45

            # Ubah gambar asli ke base64 untuk ditampilkan
            _, buffer_orig = cv2.imencode('.jpg', frame)
            original_image_b64 = base64.b64encode(buffer_orig).decode('utf-8')

            if len(all_boxes) > 0:
                for box in all_boxes:
                    x1, y1, x2, y2 = box
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # --- [TITIK RAWAN PERFORMA 2] Pengenalan Wajah ---
                    recognized_id, min_distance = _recognize_one_face_web(face_crop, embedding_db)
                    
                    status = "Tidak Dikenal"
                    nama = "N/A"
                    keyakinan = 0.0
                    
                    if recognized_id not in ["Tidak Dikenal", "Crop Gagal", "Error Proses", "DB Kosong"]:
                        keyakinan = (1 - min_distance) * 100
                        nama = embedding_db.get(recognized_id, {}).get('name', 'Error ID')
                        if min_distance < THRESHOLD_YAKIN_WEB:
                            status = "Dikenali Otomatis"
                        else:
                            status = f"Saran (Mirip {nama})"
                    
                    # Ubah face_crop ke base64
                    _, buffer_face = cv2.imencode('.jpg', face_crop)
                    face_image_b64 = base64.b64encode(buffer_face).decode('utf-8')

                    processed_results.append({
                        "status": status,
                        "confidence": f"{keyakinan:.1f}",
                        "name": nama,
                        "face_image_b64": face_image_b64
                    })
            else:
                 return render_template_string(html_template, message="Tidak ada wajah terdeteksi!", original_image_b64=original_image_b64)

            os.remove(filepath) # Hapus file sementara setelah diproses
            return render_template_string(html_template, results=processed_results, original_image_b64=original_image_b64)

    # Tampilkan halaman upload jika metode GET
    return render_template_string(html_template, results=None, message=None)

# --- Jalankan Server (Hanya untuk Development) ---
if __name__ == '__main__':
    # host='0.0.0.0' agar bisa diakses dari HP di jaringan yg sama
    # debug=True jangan dipakai di produksi
    app.run(host='0.0.0.0', port=5000, debug=True)