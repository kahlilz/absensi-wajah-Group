# Sistem Presensi Wajah dengan YOLO dan DeepFace

Prototipe sistem presensi kelas berbasis pengenalan wajah menggunakan YOLOv8n-face untuk deteksi dan SFace (via DeepFace) untuk pengenalan, dilengkapi dengan mekanisme *Active Learning* untuk meningkatkan akurasi secara berkelanjutan.

Repositori ini berisi evolusi proyek dalam tiga versi utama:

- **v1 (Original)**: Satu file Python yang mengimplementasikan fungsionalitas inti.
- **v2 (Website)**: Versi berbasis web dengan antarmuka sederhana.
- **v3 (Final)**: Aplikasi desktop yang dikemas sebagai executable (EXE) dengan struktur kode modular.

## Fitur

- **Deteksi Wajah**: Model YOLOv8n-face untuk deteksi multi-wajah akurat dalam foto grup.
- **Pengenalan Wajah**: Model SFace melalui library DeepFace untuk menghasilkan *embedding* 128-dimensi.
- ***Active Learning***: Koreksi manual pada identifikasi yang tidak pasti memperkaya *database embedding*, meningkatkan akurasi di masa mendatang.
- **Pencatatan Absensi**: Menyimpan rekaman kehadiran dalam file CSV dan *database* SQLite.
- **Antarmuka Pengguna**:
  - v1: Baris perintah / GUI sederhana (Tkinter)
  - v2: Antarmuka web (Flask)
  - v3: GUI desktop modern (CustomTkinter) dikemas sebagai EXE

## Versi

### v1 – Original (Satu File)
- Semua kode dalam satu skrip Python (`attendance_system.py`).
- GUI dasar menggunakan Tkinter.
- Fungsionalitas: unggah foto grup, deteksi wajah, cocokkan dengan *database*, koreksi manual.
- Cocok untuk menguji alur inti sistem.

### v2 – Versi Website
- Aplikasi web dibangun dengan Flask.
- Memungkinkan banyak pengguna mengunggah foto melalui peramban.
- Backend melakukan deteksi dan pengenalan, mengembalikan hasil.
- *Database* disimpan di sisi server.

### v3 – Versi Final (Modular EXE)
- Kode di-refactoring menjadi modul terpisah:
  - `face_detection.py` – Pembungkus YOLO
  - `face_recognition.py` – Ekstraksi *embedding* dan pencocokan
  - `database_manager.py` – Pengelolaan *database embedding* dan log absensi
  - `gui.py` – Antarmuka CustomTkinter
  - `active_learning.py` – Logika pembaruan *embedding*
- Dikemas sebagai aplikasi mandiri menggunakan PyInstaller.
- Tidak perlu instalasi Python; dapat dijalankan di Windows.

## Instalasi

### Prasyarat
- Python 3.8 atau lebih tinggi (untuk v1 dan v2)
- Paket yang diperlukan: lihat `requirements.txt`

### Setup
```bash
git clone https://github.com/namapengguna/sistem-presensi-wajah.git
cd sistem-presensi-wajah
pip install -r requirements.txt
```

### Menjalankan v1
```bash
python v1/attendance_system.py
```

### Menjalankan v2
```bash
cd v2
python app.py
```
Lalu buka `http://localhost:5000` di peramban Anda.

### Menjalankan v3 (EXE)
Unduh rilis terbaru dari halaman [Rilissegera](../../releases) dan jalankan `AttendanceSystem.exe`.

## Cara Penggunaan

1. **Registrasi wajah**: Unggah foto grup, beri nama pada setiap wajah yang terdeteksi. Sistem menyimpan *embedding*.
2. **Ambil absensi**: Unggah foto grup baru. Sistem mengidentifikasi wajah yang dikenal dan meminta konfirmasi untuk yang tidak pasti.
3. ***Active Learning***: Saat Anda mengoreksi identitas yang salah, *embedding* baru ditambahkan ke *database*.
4. **Lihat log**: Rekaman kehadiran disimpan dalam `attendance_log.csv` dan dapat dilihat di GUI.

## Gambaran Program
<img width="175" height="auto" alt="image" src="https://github.com/user-attachments/assets/8f9337be-290b-4797-81a6-f24574e903c5" />
<img width="175" height="auto" alt="image" src="https://github.com/user-attachments/assets/de6b090b-bc48-4c73-8aaf-78d9f73001b2" />
<img width="225" height="auto" alt="image" src="https://github.com/user-attachments/assets/1567d7d4-e34b-4ec2-859f-83ea3cd33a4f" />
<img width="225" height="auto" alt="image" src="https://github.com/user-attachments/assets/d3c9e315-f8b6-4129-b20e-dc4f4c69e04e" />


## Dataset

Sistem diuji dengan 13 foto grup (masing-masing 12 wajah) dalam dua kualitas kamera:
- Kualitas rendah (640×480)
- Kualitas baik (4624×2600)

Lihat [link menyusul](#) untuk hasil lengkap.

## Ringkasan Hasil

- Akurasi deteksi: **98,1%** (YOLOv8n-face)
- Sebelum *Active Learning* (satu *embedding* per orang): **FRR 77,8%, FAR 0%, akurasi 16,7%**
- Setelah *Active Learning* (11 foto *enrichment*, kamera kualitas baik): **FRR 11,1%, FAR 33,3%, akurasi 83,3%**

Performa sangat bergantung pada kualitas gambar dan variasi pose.

## Ucapan Terima Kasih

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [DeepFace](https://github.com/serengil/deepface)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
