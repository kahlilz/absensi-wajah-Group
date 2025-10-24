# Import library yang dibutuhkan (Flask, OpenCV, dll.)
from flask import Flask, request, jsonify, render_template_string, redirect, url_for, flash
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
import uuid # Untuk ID unik sementara

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

# --- INISIALISASI (Perlu dimuat sekali saja) ---
print("Memuat model YOLO...")
face_detector_model = YOLO(path_ke_model_str)
print("Memuat database embedding...")
embedding_db = {}
if os.path.exists(embedding_db_path):
    try:
        with open(embedding_db_path, 'rb') as f: embedding_db = pickle.load(f)
    except Exception as e: print(f"Gagal memuat DB: {e}")
print("Model dan DB siap.")

# --- Fungsi Pengenalan Wajah ---
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
        return recognized_id, min_distance, live_embedding # Kembalikan juga embedding
    except Exception as e:
        print(f"Error _recognize_one_face_web: {e}")
        return "Error Proses", float('inf'), None

# --- Setup Aplikasi Flask ---
app = Flask(__name__)
app.secret_key = 'super secret key' # Diperlukan untuk flash message
UPLOAD_FOLDER = os.path.join(APP_PATH, 'uploads_web')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- [BARU] Penyimpanan sementara untuk face_crop/embedding ---
# WARNING: Ini TIDAK aman untuk produksi (bisa penuh, tidak thread-safe)
# Hanya untuk DEMO sederhana.
temp_face_data = {}

# --- [MODIFIKASI] Halaman Utama HTML ---
html_template = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Absen Wajah Web (Demo)</title>
    <style> 
        body { font-family: sans-serif; padding: 20px; } 
        li { border: 1px solid #ccc; margin-bottom: 15px; padding: 10px; list-style: none; }
        .face-info { display: inline-block; vertical-align: top; margin-left: 15px; }
        select, button { padding: 5px; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>Demo Absensi Wajah via Web</h1>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul style="color: green;">
        {% for message in messages %}
          <li>{{ message }}</li>
        {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}
    
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
                <img src="data:image/jpeg;base64,{{ result.face_image_b64 }}" alt="Face" style="width: 100px; height: auto; vertical-align: top;">
                <div class="face-info">
                    {% if result.is_confident %}
                        Status: {{ result.status }} (Keyakinan: {{ result.confidence }}%) - Nama: {{ result.name }}<br>
                        <strong style="color: green;">✅ Absen Otomatis Tercatat</strong>
                    {% else %}
                        Status: {{ result.status }} (Keyakinan: {{ result.confidence }}%)<br>
                        Nama Saran: {{ result.name }}<br>
                        <!-- Form Koreksi -->
                        <form action="{{ url_for('confirm_attendance') }}" method="post">
                            <input type="hidden" name="temp_face_id" value="{{ result.temp_id }}">
                            <label for="nama_koreksi_{{ loop.index }}">Pilih Nama Benar:</label>
                            <select name="selected_user_id" id="nama_koreksi_{{ loop.index }}">
                                <option value="Tidak Dikenal">Tidak Dikenal</option>
                                {% for name, user_id in name_options.items() %}
                                    {% if name != 'Tidak Dikenal' %}
                                        <option value="{{ user_id }}" {% if name == result.name %}selected{% endif %}>{{ name }}</option>
                                    {% endif %}
                                {% endfor %}
                            </select><br>
                            <button type="submit">Konfirmasi & Catat</button>
                        </form>
                    {% endif %}
                </div>
            </li>
        {% endfor %}
        </ul>
    {% elif message %}
        <p style="color: red;">{{ message }}</p>
    {% endif %}
</body>
</html>
"""

# --- [MODIFIKASI] Endpoint Utama ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    global temp_face_data # Akses penyimpanan sementara
    if request.method == 'POST':
        # ... (Kode upload & baca file sama seperti sebelumnya) ...
        if 'file' not in request.files: return render_template_string(html_template, message="Tidak ada file dipilih!", name_options=get_name_options())
        file = request.files['file']
        if file.filename == '': return render_template_string(html_template, message="Nama file kosong!", name_options=get_name_options())
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            frame = cv2.imread(filepath)
            if frame is None:
                os.remove(filepath) 
                return render_template_string(html_template, message="Gagal membaca file gambar!", name_options=get_name_options())

            results_detection = face_detector_model(frame, verbose=False, conf=0.4)
            all_boxes = results_detection[0].boxes.xyxy.cpu().numpy().astype(int)
            processed_results = []
            THRESHOLD_YAKIN_WEB = 0.45
            temp_face_data.clear() # Kosongkan data lama

            _, buffer_orig = cv2.imencode('.jpg', frame)
            original_image_b64 = base64.b64encode(buffer_orig).decode('utf-8')

            if len(all_boxes) > 0:
                for i, box in enumerate(all_boxes):
                    x1, y1, x2, y2 = box
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # --- [TITIK RAWAN PERFORMA] ---
                    recognized_id, min_distance, live_embedding = _recognize_one_face_web(face_crop, embedding_db)
                    
                    is_confident = False
                    status = "Tidak Dikenal"
                    nama = "N/A"
                    keyakinan = 0.0
                    temp_id = str(uuid.uuid4()) # ID unik untuk wajah ini

                    if recognized_id not in ["Tidak Dikenal", "Crop Gagal", "Error Proses", "DB Kosong"]:
                        keyakinan = (1 - min_distance) * 100
                        nama = embedding_db.get(recognized_id, {}).get('name', 'Error ID')
                        if min_distance < THRESHOLD_YAKIN_WEB:
                            status = "Dikenali Otomatis"
                            is_confident = True
                            # Catat absen otomatis
                            mark_attendance_web(nama, recognized_id, face_crop)
                        else:
                            status = f"Saran (Mirip {nama})"
                            # Simpan data untuk konfirmasi nanti
                            temp_face_data[temp_id] = {'crop': face_crop, 'embedding': live_embedding}
                    else:
                         # Simpan data untuk konfirmasi nanti (jika ingin bisa memilih 'Tidak Dikenal')
                         temp_face_data[temp_id] = {'crop': face_crop, 'embedding': live_embedding}

                    _, buffer_face = cv2.imencode('.jpg', face_crop)
                    face_image_b64 = base64.b64encode(buffer_face).decode('utf-8')

                    processed_results.append({
                        "status": status,
                        "confidence": f"{keyakinan:.1f}",
                        "name": nama,
                        "face_image_b64": face_image_b64,
                        "is_confident": is_confident,
                        "temp_id": temp_id # Kirim ID sementara ke template
                    })
            else:
                 return render_template_string(html_template, message="Tidak ada wajah terdeteksi!", original_image_b64=original_image_b64, name_options=get_name_options())

            os.remove(filepath) 
            return render_template_string(html_template, results=processed_results, original_image_b64=original_image_b64, name_options=get_name_options())

    # Tampilkan halaman upload jika metode GET
    return render_template_string(html_template, results=None, message=None, name_options=get_name_options())

# --- [BARU] Endpoint untuk konfirmasi ---
@app.route('/confirm', methods=['POST'])
def confirm_attendance():
    global embedding_db, temp_face_data # Perlu akses global

    temp_face_id = request.form.get('temp_face_id')
    selected_user_id_str = request.form.get('selected_user_id')

    if not temp_face_id or temp_face_id not in temp_face_data:
        flash("Error: Data wajah sementara tidak ditemukan.")
        return redirect(url_for('upload_and_process'))
        
    if not selected_user_id_str or selected_user_id_str == "Tidak Dikenal":
        flash("Konfirmasi: Wajah ditandai sebagai 'Tidak Dikenal'.")
        del temp_face_data[temp_face_id] # Hapus data sementara
        return redirect(url_for('upload_and_process'))

    try:
        selected_user_id = int(selected_user_id_str)
        if selected_user_id not in embedding_db:
             flash(f"Error: ID User {selected_user_id} tidak ada di database.")
             return redirect(url_for('upload_and_process'))

        nama_terpilih = embedding_db[selected_user_id]['name']
        face_data = temp_face_data[temp_face_id]
        face_crop = face_data['crop']
        new_embedding = face_data['embedding'] # Ambil embedding yg sudah dihitung

        # 1. Catat Absen
        mark_attendance_web(nama_terpilih, selected_user_id, face_crop)

        # 2. Active Learning
        if new_embedding is not None:
             if 'embeddings' in embedding_db[selected_user_id]:
                 embedding_db[selected_user_id]['embeddings'].append(new_embedding)
             else: # Jika masih format lama
                 old_embedding = embedding_db[selected_user_id]['embedding']
                 embedding_db[selected_user_id]['embeddings'] = [old_embedding, new_embedding]
                 del embedding_db[selected_user_id]['embedding']
             
             # Simpan DB ke file
             save_embedding_db_web()
             print(f"INFO (Web): DB untuk '{nama_terpilih}' diperbarui.")
             flash(f"Absen untuk {nama_terpilih} tercatat & database diperbarui!")
        else:
             flash(f"Absen untuk {nama_terpilih} tercatat (DB tidak diupdate - embedding error).")

        del temp_face_data[temp_face_id] # Hapus data sementara setelah diproses

    except ValueError:
         flash("Error: ID User yang dipilih tidak valid.")
    except Exception as e:
         flash(f"Error saat konfirmasi: {e}")
         print(f"ERROR saat konfirmasi web: {e}")

    return redirect(url_for('upload_and_process')) # Kembali ke halaman utama

# --- Helper Functions (Web Version) ---
def get_name_options():
    """Membuat dictionary nama -> ID untuk dropdown"""
    options = {"Tidak Dikenal": "Tidak Dikenal"}
    for user_id, data in embedding_db.items():
        options[data['name']] = user_id
    return options

def mark_attendance_web(name, user_id, face_crop):
    """Versi web untuk mencatat absen & simpan foto log"""
    try:
        log_photo_dir = os.path.join(APP_PATH, "log_absensi", "foto_log")
        os.makedirs(log_photo_dir, exist_ok=True)
        now = datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S') 
        filename = f"{user_id}_{timestamp_str}.jpg"
        photo_path = os.path.join(log_photo_dir, filename)
        cv2.imwrite(photo_path, face_crop) # Simpan foto bukti

        file_path_csv = os.path.join(APP_PATH, "log_absensi", "Attendance.csv")
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        file_exists = os.path.isfile(file_path_csv)
        
        with open(file_path_csv, 'a', newline='', encoding='utf-8') as f:
            if not file_exists: f.write("Nama;ID Siswa;Tanggal;Waktu;Path Foto\n")
            f.write(f"{name};{user_id};{date_str};{time_str};{photo_path}\n")
        print(f"Absen Dicatat (Web): {name}")
    except Exception as e:
        print(f"Error mark_attendance_web: {e}")

def save_embedding_db_web():
    """Versi web untuk menyimpan DB"""
    global embedding_db
    try:
        with open(embedding_db_path, 'wb') as f:
            pickle.dump(embedding_db, f)
    except Exception as e:
        print(f"Error save_embedding_db_web: {e}")

# --- Jalankan Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
