# Smart Attendance System (YOLOv8 + SFace + Active Learning)

Sistem Absensi Wajah Adaptif adalah aplikasi desktop Python yang dirancang untuk melakukan absensi dengan memproses foto grup. Proyek ini menggunakan model AI modern untuk deteksi dan pengenalan wajah, dengan fokus utama pada kemampuan sistem untuk belajar dan beradaptasi dari waktu ke waktu.

## Fitur Utama

- Deteksi Multi-Wajah: Menggunakan YOLOv8-Face untuk mendeteksi semua wajah dalam satu gambar grup secara efisien.

- Pengenalan Akurat: Menggunakan model SFace (via DeepFace) untuk menghasilkan embedding wajah yang andal.

- Absensi Adaptif (Active Learning): Fitur inti dari sistem ini.

  - Absen Otomatis: Wajah dengan tingkat keyakinan tinggi (misal, > 55%) akan diabsen secara otomatis.

  - Menu Review Interaktif: Wajah dengan keyakinan sedang atau tidak dikenal akan disajikan kepada pengguna dalam menu review.

  - Pembelajaran dari Koreksi: Pengguna dapat mengoreksi sistem dengan memilih nama yang benar dari dropdown. Saat  dikonfirmasi, embedding dari wajah baru ini ditambahkan ke database, sehingga sistem menjadi lebih "pintar" dan akurat dalam mengenali orang tersebut di masa depan.

- Manajemen Database: Antarmuka (GUI) untuk melihat semua pengguna yang terdaftar, jumlah foto/embedding yang tersimpan, dan menghapus data pengguna.

- Log Audit: Setiap absensi yang tercatat akan disimpan dalam file Attendance.csv dan juga menyimpan foto crop wajah sebagai bukti di folder log_absensi/foto_log/.
```bash
pip install foobar
```

## Visual

Berikut adalah beberapa tampilan antarmuka aplikasi:

1. Menu Utama: Tempat navigasi utama untuk Pendaftaran, Absensi, dan Manajemen. 
<img width="340" height="549" alt="{220DFF5B-7A57-4F02-A632-BBB860485CAE}" src="https://github.com/user-attachments/assets/21b3b267-7189-4072-b121-6dad82603920" />

2. Proses Pendaftaran: Setelah mengunggah foto, sistem mendeteksi semua wajah dan menyediakan kolom untuk mendaftarkan Nama dan ID. 
<img width="340" height="549" alt="{52CE0E51-A664-4DB7-BDED-C826FA992B03}" src="https://github.com/user-attachments/assets/a562c1f8-557c-4ba3-a9f0-bb2700b838a3" />

3. Menu Review Absensi (Inti Fitur): Menampilkan semua wajah yang terdeteksi dari foto kelas, lengkap dengan status (Otomatis/Saran) dan dropdown untuk koreksi manual. 
<img width="452" height="548" alt="{2955CEF3-E573-4042-971B-2E7BC2121343}" src="https://github.com/user-attachments/assets/d73613f7-42c4-4abf-b7cd-06fae07369f8" />

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Instalasi

Ada dua cara untuk menjalankan aplikasi ini: menggunakan file .exe yang sudah jadi atau menjalankan dari kode sumber.

**Opsi 1: Menjalankan Aplikasi (.exe)**

Ini adalah cara yang direkomendasikan untuk pengguna biasa.

1. Buka halaman Releases di repositori ini.
2. Unduh file .zip versi terbaru (misal SmartAttendance_v1.0.zip).
3. Ekstrak file .zip tersebut ke dalam sebuah folder baru.
4. Jalankan file .exe (misal Yolov8New.exe atau smart_attendance_system.exe).
5. Saat pertama kali dijalankan, aplikasi akan secara otomatis membuat folder database dan log_absensi di dalam folder yang sama.

**Opsi 2: Menjalankan dari Kode Sumber (Untuk Developer)**

Persyaratan:

Python 3.11 atau lebih baru.

git (opsional, untuk kloning).

File model YOLOv8-Face: yolov8n-face-lindevs.pt.

Langkah-langkah:

Kloning repositori ini (atau unduh sebagai ZIP):

git clone [https://github.com/username/project-name.git](https://github.com/username/project-name.git)
cd project-name


Buat dan aktifkan virtual environment:

python -m venv venv
source venv/bin/activate  # Untuk Windows: venv\Scripts\activate


Instal library yang dibutuhkan:

pip install -r requirements.txt


(Jika requirements.txt tidak ada, instal manual: pip install customtkinter ultralytics deepface opencv-python pillow scipy)

PENTING: Unduh file model yolov8n-face-lindevs.pt dan letakkan di dalam folder proyek utama (folder yang sama dengan file .py Anda).

Jalankan aplikasi:

python smart_attendance_system.py


Cara Penggunaan

Pendaftaran Awal (Hanya sekali per orang):

Klik tombol "Unggah Foto untuk Pendaftaran".

Pilih foto yang berisi wajah-wajah orang yang ingin Anda daftarkan.

Untuk setiap wajah yang terdeteksi, isi Nama Lengkap dan ID Siswa (harus berupa angka).

Klik "Simpan & Tambah ke Database". Database awal Anda kini telah dibuat.

Melakukan Absensi (Proses Inti):

Klik tombol "Unggah & Proses Foto Kelas".

Pilih foto grup (foto kelas) yang ingin Anda proses.

Aplikasi akan memproses dan menampilkan "Menu Review Absensi" di jendela baru.

Periksa Hasil:

Wajah yang dikenali otomatis akan ditandai "✅ Absen Otomatis Tercatat".

Wajah yang ragu-ragu atau tidak dikenal akan menampilkan dropdown "Pilih Nama Benar:".

Lakukan Koreksi (Active Learning):

Untuk setiap wajah yang salah atau tidak dikenal, pilih nama yang benar dari dropdown.

Klik tombol "Konfirmasi & Catat Absen".

Saat Anda mengonfirmasi, sistem akan mencatat absensi DAN memperbarui database dengan embedding baru dari wajah tersebut, membuat sistem lebih pintar untuk ke depannya.

Mengelola Database:

Klik tombol "Lihat & Kelola Database".

Jendela baru akan menampilkan semua pengguna yang terdaftar, ID mereka, dan jumlah foto (embedding) yang tersimpan.

Anda dapat menghapus pengguna dari database menggunakan tombol "Hapus".

## Roadmap

Proyek ini adalah prototipe fungsional. Ide untuk pengembangan di masa depan meliputi:

Versi Web/Mobile: Membangun backend (Flask/Django) agar sistem dapat diakses melalui browser handphone, seperti yang dieksplorasi selama penelitian.

Deteksi Liveness: Menambahkan modul anti-spoofing untuk memastikan wajah yang diabsen adalah orang sungguhan, bukan foto atau video.

Database Terpusat: Migrasi dari file .pkl lokal ke database SQL (seperti PostgreSQL) untuk skalabilitas yang lebih baik.

Kontribusi

Proyek ini adalah bagian dari penelitian skripsi, namun masukan selalu diterima. Jika Anda menemukan bug atau memiliki ide, silakan buka Issue.

Lisensi

Proyek ini dilisensikan di bawah MIT License. Lihat file LICENSE untuk detailnya.

## License

[MIT](https://choosealicense.com/licenses/mit/)




<img width="378" height="474" alt="{10A751EC-5343-4ED3-B04A-3470F8498D4B}" src="https://github.com/user-attachments/assets/4525569d-4b07-497c-ad19-f2ec88a71bcb" />

test
