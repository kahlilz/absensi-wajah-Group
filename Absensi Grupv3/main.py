import customtkinter as ctk
import cv2
import os
import sys
import numpy as np
import time # Diperlukan untuk jeda
from PIL import Image
from tkinter import filedialog, messagebox # messagebox untuk konfirmasi
from datetime import datetime
import pickle
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine
import uuid

# --- [FUNGSI KUNCI] Menentukan Path Aplikasi ---
def get_app_path():
    """Mendapatkan path root aplikasi, berfungsi untuk mode dev dan .exe."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path):
    """ Mendapatkan path absolut ke file sumber daya (seperti model .pt)."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- [MODIFIKASI] Gunakan path aplikasi sebagai basis ---
APP_PATH = get_app_path()

# Pengecekan Folder, sekarang dibuat relatif terhadap lokasi aplikasi
os.makedirs(os.path.join(APP_PATH, 'database'), exist_ok=True)
os.makedirs(os.path.join(APP_PATH, 'log_absensi'), exist_ok=True)
os.makedirs(os.path.join(APP_PATH, 'log_absensi', 'foto_log'), exist_ok=True)

# Path model menggunakan fungsi resource_path
path_ke_model_str = resource_path('yolov8n-face-lindevs.pt')
face_detector_model = YOLO(path_ke_model_str)

# Path untuk database embedding sekarang juga absolut
embedding_db_path = os.path.join(APP_PATH, 'database', 'embeddings.pkl')


class SmartAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- KONFIGURASI JENDELA & TEMA ---
        self.title("Smart Attendance [YOLO + SFace]")
        self.geometry("450x700")
        self.resizable(True, True)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # --- VARIABEL INSTANCE ---
        self.camera_index = 0 # Default camera index
        self.detected_faces_data = []
        self.embedding_db = self.load_embedding_db()
        self.db_window = None # Untuk melacak jendela manajemen DB
        self.timer_job_id = None # Untuk melacak ID job timer
        self.current_repetition = 0
        self.total_repetitions = 0
        self.timer_interval_ms = 0

        # --- FRAME UTAMA ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#242630")
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.title_label = ctk.CTkLabel(self.main_frame, text="Smart Attendance",
                                        font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(10, 10))

        self.status_label = ctk.CTkLabel(self.main_frame, text="Selamat Datang!", wraplength=380,
                                         font=ctk.CTkFont(size=14))
        self.status_label.pack(side="bottom", pady=10, padx=10, fill="x")

        self.master_scroll_frame = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        self.master_scroll_frame.pack(pady=5, padx=10, fill="both", expand=True)

        # --- [MODIFIKASI] Pilihan Input (Checkbox) ---
        input_option_frame = ctk.CTkFrame(self.master_scroll_frame, fg_color="transparent")
        input_option_frame.pack(fill="x", padx=10, pady=5)
        self.use_camera_var = ctk.BooleanVar(value=False) # Default ke upload file
        self.use_camera_checkbox = ctk.CTkCheckBox(input_option_frame, text="Gunakan Kamera (untuk Daftar & Absen Timer)",
                                                   variable=self.use_camera_var, font=ctk.CTkFont(size=14),
                                                   command=self._toggle_input_mode) # Tambah command
        self.use_camera_checkbox.pack(pady=(5, 10), anchor="w")

        # --- BAGIAN 1: PENGAMBILAN DATA WAJAH ---
        self.input_method_label = ctk.CTkLabel(self.master_scroll_frame, text="1. Tambah Data Wajah",
                                               font=ctk.CTkFont(size=16, weight="bold"))
        self.input_method_label.pack(pady=(5, 5), anchor="w", padx=10)
        self.tambah_data_button = ctk.CTkButton(self.master_scroll_frame, text="Unggah Foto Pendaftaran", # Initial text
                                                height=40, command=self.handle_tambah_data)
        self.tambah_data_button.pack(pady=5, fill="x", padx=10)

        self.preview_frame_container = ctk.CTkFrame(self.master_scroll_frame, fg_color="transparent")

        # --- BAGIAN 2: ABSENSI ---
        self.attend_label = ctk.CTkLabel(self.master_scroll_frame, text="2. Mulai Absensi",
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.attend_label.pack(pady=(20, 5), anchor="w", padx=10)
        
        opsi_absen_frame = ctk.CTkFrame(self.master_scroll_frame, fg_color="transparent")
        opsi_absen_frame.pack(fill="x", padx=10, pady=5)
        
        # Tombol Absen via Unggah Foto (selalu ada)
        self.upload_absen_button = ctk.CTkButton(opsi_absen_frame, text="Absen via Unggah Foto", height=40,
                                                  command=self.absensiDariGambar)
        self.upload_absen_button.pack(fill="x", pady=(0, 10))

        # --- Opsi Timer (hanya muncul jika checkbox dicentang) ---
        self.timer_frame = ctk.CTkFrame(opsi_absen_frame)
        # timer_frame dipack/unpack oleh _toggle_input_mode

        timer_label = ctk.CTkLabel(self.timer_frame, text="Absensi Otomatis via Timer (Gunakan Kamera):")
        timer_label.pack(anchor="w", padx=10, pady=(5,0))

        input_frame = ctk.CTkFrame(self.timer_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        interval_label = ctk.CTkLabel(input_frame, text="Interval (menit):")
        interval_label.pack(side="left", padx=(0, 5))
        self.interval_entry = ctk.CTkEntry(input_frame, width=60, placeholder_text="10")
        self.interval_entry.pack(side="left", padx=5)
        self.interval_entry.insert(0, "10")

        repeat_label = ctk.CTkLabel(input_frame, text="Ulangi (kali):")
        repeat_label.pack(side="left", padx=(10, 5))
        self.repeat_entry = ctk.CTkEntry(input_frame, width=60, placeholder_text="2")
        self.repeat_entry.pack(side="left", padx=5)
        self.repeat_entry.insert(0, "2")

        # --- [MODIFIKASI] Pemilihan Kamera untuk Timer ---
        self.camera_combobox = ctk.CTkComboBox(self.timer_frame, height=35, command=self.set_camera_index)
        self.camera_combobox.pack(pady=(10, 5), fill="x", padx=10)
        self.detect_cameras() # Panggil detect cameras di init

        self.start_timer_button = ctk.CTkButton(self.timer_frame, text="Mulai Absen Terjadwal Kamera", height=35,
                                                 command=self.mulai_absensi_terjadwal_kamera)
        self.start_timer_button.pack(fill="x", padx=10, pady=(5, 10))

        # --- BAGIAN 3: MANAJEMEN DATABASE ---
        self.manage_label = ctk.CTkLabel(self.master_scroll_frame, text="3. Manajemen Database",
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.manage_label.pack(pady=(20, 5), anchor="w", padx=10)
        self.view_db_button = ctk.CTkButton(self.master_scroll_frame, text="Lihat & Kelola Database", height=40,
                                              fg_color="#1F6AA5", hover_color="#144E7A",
                                              command=self.tampilkan_database)
        self.view_db_button.pack(pady=5, fill="x", padx=10)

        # Panggil toggle mode di akhir __init__ untuk setup awal UI
        self._toggle_input_mode()

    # --- [BARU] Fungsi untuk mengatur visibilitas UI berdasarkan checkbox ---
    def _toggle_input_mode(self):
        use_camera = self.use_camera_var.get()
        if use_camera:
            self.tambah_data_button.configure(text="Ambil Foto Pendaftaran via Kamera", command=self.ambil_dari_kamera)
            self.timer_frame.pack(fill="x", pady=5) # Tampilkan frame timer
            self.detect_cameras() # Pastikan daftar kamera ada
            if self.camera_index == -1: # Nonaktifkan jika tidak ada kamera
                 self.start_timer_button.configure(state="disabled")
            else:
                 self.start_timer_button.configure(state="normal")
        else:
            self.tambah_data_button.configure(text="Unggah Foto Pendaftaran", command=self.unggah_gambar_daftar)
            self.timer_frame.pack_forget() # Sembunyikan frame timer

    # --- FUNGSI DISPATCHER (PENGATUR) ---
    def handle_tambah_data(self):
        # Fungsi ini sekarang dikontrol oleh command button yang diubah oleh _toggle_input_mode
        # Cukup panggil command yang sudah terpasang
        current_command = self.tambah_data_button.cget("command")
        if current_command:
             current_command()

    # Fungsi ini tidak lagi diperlukan karena tombol absen timer terpisah
    # def handle_mulai_absensi(self):
    #     pass

    # --- Fungsi Kamera & UI ---
    def detect_cameras(self):
        available_cameras = []
        try:
            # Coba deteksi hingga 5 kamera
            for i in range(5):
                cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
                if cap.isOpened():
                    available_cameras.append(f"Kamera {i}")
                    cap.release() # Penting: lepaskan kamera setelah dicek
                else:
                    break # Berhenti jika indeks sudah tidak valid
        except Exception as e:
            print(f"Error saat deteksi kamera: {e}")

        if available_cameras:
            self.camera_combobox.configure(values=available_cameras, state="normal")
            # Coba set ke kamera yang terakhir dipilih jika masih ada
            current_selection = f"Kamera {self.camera_index}"
            if current_selection in available_cameras:
                 self.camera_combobox.set(current_selection)
            else:
                 self.camera_combobox.set(available_cameras[0])
                 self.camera_index = 0 # Reset jika kamera lama hilang
            # Aktifkan tombol timer jika mode kamera aktif
            if self.use_camera_var.get():
                 self.start_timer_button.configure(state="normal")

        else:
            self.camera_combobox.configure(values=["Tidak Ada Kamera"], state="disabled")
            self.camera_combobox.set("Tidak Ada Kamera")
            self.camera_index = -1
            self.status_label.configure(text="Tidak ada kamera terdeteksi!", text_color="orange")
            # Nonaktifkan tombol timer
            self.start_timer_button.configure(state="disabled")


    def set_camera_index(self, choice):
        try:
            # Pastikan choice valid sebelum parsing
            if "Kamera" in choice:
                 self.camera_index = int(choice.split(' ')[1])
                 self.status_label.configure(text=f"{choice} terpilih.", text_color="white")
                 if self.use_camera_var.get(): # Aktifkan tombol timer jika valid
                      self.start_timer_button.configure(state="normal")
            else:
                 self.camera_index = -1
                 self.start_timer_button.configure(state="disabled") # Nonaktifkan jika tidak valid
        except (ValueError, IndexError):
            self.camera_index = -1
            self.start_timer_button.configure(state="disabled")

    # --- FUNGSI BARU UNTUK ABSENSI TERJADWAL VIA KAMERA ---
    def mulai_absensi_terjadwal_kamera(self):
        if self.camera_index == -1:
             self.status_label.configure(text="Pilih kamera yang valid terlebih dahulu.", text_color="orange")
             return
        if not self.embedding_db:
             self.status_label.configure(text="Database kosong. Tambah data wajah dulu.", text_color="orange")
             return

        try:
            interval_menit = int(self.interval_entry.get())
            self.total_repetitions = int(self.repeat_entry.get())
            if interval_menit <= 0 or self.total_repetitions <= 0:
                raise ValueError("Interval dan repetisi harus > 0")
        except ValueError:
            self.status_label.configure(text="Input interval/repetisi tidak valid (harus angka > 0).", text_color="orange")
            return

        self.timer_interval_ms = interval_menit * 60 * 1000
        self.current_repetition = 0

        self.start_timer_button.configure(state="disabled", text="Timer Kamera Berjalan...")
        self.upload_absen_button.configure(state="disabled")
        self.tambah_data_button.configure(state="disabled")
        self.use_camera_checkbox.configure(state="disabled") # Nonaktifkan checkbox juga

        self.status_label.configure(text=f"Absensi kamera dimulai. Menunggu {interval_menit} menit...", text_color="cyan")
        self._jalankan_timer_kamera() # Ganti nama fungsi

    def _jalankan_timer_kamera(self):
        if self.current_repetition < self.total_repetitions:
            self.current_repetition += 1
            status_msg = f"Menunggu interval {self.current_repetition}/{self.total_repetitions} ({int(self.timer_interval_ms/60000)} menit)..."
            self.status_label.configure(text=status_msg, text_color="cyan")
            self.timer_job_id = self.after(self.timer_interval_ms, self._capture_and_process_kamera) # Ganti nama fungsi
        else:
            self._selesai_timer_kamera() # Ganti nama fungsi

    def _capture_and_process_kamera(self):
        self.status_label.configure(text=f"Mengambil gambar ({self.current_repetition}/{self.total_repetitions}) dari Kamera {self.camera_index}...", text_color="yellow")
        self.update_idletasks()
        
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        if not cap.isOpened():
            self.status_label.configure(text=f"Error: Gagal membuka Kamera {self.camera_index}.", text_color="red")
            self._selesai_timer_kamera()
            return

        time.sleep(1)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            self.status_label.configure(text=f"Error: Gagal mengambil gambar dari Kamera {self.camera_index}.", text_color="red")
            self._selesai_timer_kamera()
            return

        self.status_label.configure(text=f"Memproses gambar ({self.current_repetition}/{self.total_repetitions})...", text_color="cyan")
        self.update_idletasks()

        self._proses_gambar_untuk_review(frame)

        # Jadwalkan timer berikutnya setelah proses review (jika jendela ditutup)
        # Kita asumsikan pengguna menutup jendela review sebelum interval berikutnya
        if self.current_repetition < self.total_repetitions:
            self._jalankan_timer_kamera()
        else:
            self._selesai_timer_kamera()

    def _selesai_timer_kamera(self):
        if self.timer_job_id:
            self.after_cancel(self.timer_job_id)
            self.timer_job_id = None
        
        self.start_timer_button.configure(state="normal", text="Mulai Absen Terjadwal Kamera")
        self.upload_absen_button.configure(state="normal")
        self.tambah_data_button.configure(state="normal")
        self.use_camera_checkbox.configure(state="normal") # Aktifkan checkbox lagi
        
        if self.current_repetition >= self.total_repetitions:
             self.status_label.configure(text="Absensi terjadwal selesai.", text_color="lightgreen")

    # --- FUNGSI AMBIL GAMBAR DARI KAMERA (UNTUK PENDAFTARAN) ---
    def ambil_dari_kamera(self):
        if self.camera_index == -1:
             self.status_label.configure(text="Pilih kamera yang valid terlebih dahulu.", text_color="orange")
             return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        if not cap.isOpened():
            self.status_label.configure(text=f"Error: Gagal membuka Kamera {self.camera_index}.", text_color="red")
            return

        window_name = f"Kamera {self.camera_index} - Tekan 'S' Simpan, 'Q' Keluar"
        cv2.namedWindow(window_name)

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Deteksi visual saja, tidak perlu proses embedding di sini
            results_viz = face_detector_model(frame, stream=True, verbose=False, conf=0.5)
            frame_disp = frame.copy() # Salin frame untuk display
            for r in results_viz:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow(window_name, frame_disp)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.process_and_show_faces(frame) # Kirim frame asli (tanpa kotak) untuk diproses
                break
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        # Refresh daftar kamera setelah window ditutup, jaga-jaga jika ada perubahan
        self.detect_cameras()


    # --- (Fungsi _proses_gambar_untuk_review, absensiDariGambar, _handle_konfirmasi_absen_gambar, _recognize_one_face, dll. tetap sama) ---
    # Pastikan _proses_gambar_untuk_review TIDAK memanggil _jalankan_timer atau _selesai_timer
    def _proses_gambar_untuk_review(self, frame):
        """Memproses satu frame gambar dan menampilkan menu review."""
        THRESHOLD_YAKIN_GAMBAR = 0.45

        results = face_detector_model(frame, verbose=False, conf=0.4)
        all_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        if len(all_boxes) == 0:
            self.status_label.configure(text="Tidak ada wajah terdeteksi pada gambar ini.", text_color="orange")
            # Jangan panggil _selesai_timer di sini jika dipanggil dari upload manual
            return

        konfirmasi_window = ctk.CTkToplevel(self)
        konfirmasi_window.title("Menu Review Absensi")
        konfirmasi_window.geometry("600x700")
        konfirmasi_window.transient(self)
        konfirmasi_window.attributes('-topmost', True) # Agar jendela review muncul di depan

        scroll_frame = ctk.CTkScrollableFrame(konfirmasi_window, label_text="Hasil Deteksi - Konfirmasi Absensi di Bawah")
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        pilihan_nama = {"Tidak Dikenal": "Tidak Dikenal"}
        db_items = list(self.embedding_db.items()) # Salin item DB untuk iterasi aman
        for user_id, data in db_items:
             # Tambahkan pengecekan tipe data untuk nama
             nama = data.get('name')
             if isinstance(nama, str):
                  pilihan_nama[nama] = user_id
             else:
                  print(f"Peringatan: Nama tidak valid (bukan string) untuk ID {user_id}")


        absen_otomatis_count = 0

        for i, box in enumerate(all_boxes):
            x1, y1, x2, y2 = box
            # Pastikan koordinat valid sebelum crop
            y1, y2 = max(0, y1), min(frame.shape[0], y2)
            x1, x2 = max(0, x1), min(frame.shape[1], x2)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                 print(f"Peringatan: Ukuran face_crop 0 untuk box {i}. Melewati.")
                 continue

            recognized_id, min_distance = self._recognize_one_face(face_crop)

            card_frame = ctk.CTkFrame(scroll_frame, border_width=1)
            card_frame.pack(fill="x", padx=10, pady=10)

            try:
                 face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                 ctk_image = ctk.CTkImage(light_image=face_pil, dark_image=face_pil, size=(100, 100))
                 preview_label = ctk.CTkLabel(card_frame, image=ctk_image, text="")
                 preview_label.pack(side="left", padx=10, pady=10)
            except Exception as e:
                 print(f"Error saat membuat preview gambar: {e}")
                 preview_label = ctk.CTkLabel(card_frame, text="Gagal Load Preview", width=100, height=100)
                 preview_label.pack(side="left", padx=10, pady=10)


            info_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
            info_frame.pack(side="left", fill="x", expand=True, padx=10)

            is_confident = False
            nama_dikenali = "N/A"
            if recognized_id not in ["Tidak Dikenal", "Crop Gagal", "Error Proses", "DB Kosong"]:
                 # Pastikan recognized_id ada di DB sebelum cek keyakinan
                 if recognized_id in self.embedding_db:
                      if min_distance < THRESHOLD_YAKIN_GAMBAR:
                           is_confident = True
                      nama_dikenali = self.embedding_db.get(recognized_id, {}).get('name', 'Error ID')
                 else:
                      # Jika ID tidak ada di DB (kasus aneh), anggap tidak dikenal
                      recognized_id = "Tidak Dikenal"
                      min_distance = float('inf')


            if is_confident:
                self.mark_attendance(nama_dikenali, recognized_id, face_crop)
                absen_otomatis_count += 1

                keyakinan = (1 - min_distance) * 100
                info_text = f"Nama: {nama_dikenali}\nID: {recognized_id}\nKeyakinan: {keyakinan:.1f}%"
                info_label = ctk.CTkLabel(info_frame, text=info_text, justify="left")
                info_label.pack(anchor="w", pady=5)

                status_label = ctk.CTkLabel(info_frame, text="✅ Absen Otomatis Tercatat", font=ctk.CTkFont(weight="bold"), text_color="lightgreen")
                status_label.pack(anchor="w", pady=10)
            else:
                saran_sistem = "Saran: Tidak Dikenal"
                nama_saran = "Tidak Dikenal"
                keyakinan = 0.0 # Default jika tidak dikenal
                if recognized_id not in ["Tidak Dikenal", "Crop Gagal", "Error Proses", "DB Kosong"]:
                    nama_saran = nama_dikenali
                    keyakinan = (1 - min_distance) * 100
                    saran_sistem = f"Saran: {nama_saran} (Yakin: {keyakinan:.0f}%)"

                saran_label = ctk.CTkLabel(info_frame, text=saran_sistem, font=ctk.CTkFont(size=12))
                saran_label.pack(anchor="w", pady=(5,0))

                # Pastikan pilihan_nama.keys() aman
                nama_combobox = ctk.CTkComboBox(info_frame, values=list(pilihan_nama.keys()), width=200)
                # Set default ke saran, atau 'Tidak Dikenal' jika saran tidak valid
                nama_combobox.set(nama_saran if nama_saran in pilihan_nama else "Tidak Dikenal")
                nama_combobox.pack(anchor="w", pady=5)

                konfirmasi_button = ctk.CTkButton(info_frame, text="Konfirmasi & Catat Absen")
                konfirmasi_button.pack(anchor="w", pady=10)
                konfirmasi_button.configure(command=lambda nc=nama_combobox, pn=pilihan_nama, btn=konfirmasi_button, fc=face_crop.copy(): # Salin face_crop
                    self._handle_konfirmasi_absen_gambar(pn.get(nc.get()), nc.get(), btn, fc))

        status_text_akhir = f"Review Selesai! {absen_otomatis_count} absen otomatis tercatat."
        # Update status di jendela utama hanya jika dipanggil dari thread utama
        if self.timer_job_id is None: # Indikasi dipanggil dari upload manual
             self.status_label.configure(text=status_text_akhir, text_color="white")
        else: # Dipanggil dari timer, print saja
             print(status_text_akhir)

        close_button = ctk.CTkButton(konfirmasi_window, text="Tutup Review", command=konfirmasi_window.destroy)
        close_button.pack(pady=10)


    def absensiDariGambar(self):
        filepath = filedialog.askopenfilename(
            title="Pilih Gambar untuk Absensi Manual",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath: return

        frame = cv2.imread(filepath)
        if frame is None:
            self.status_label.configure(text="Gagal membaca file gambar.", text_color="red")
            return

        self.status_label.configure(text="Memproses gambar unggahan...", text_color="cyan")
        self.update_idletasks()
        
        # Panggil fungsi pemrosesan inti
        self._proses_gambar_untuk_review(frame)

    def _handle_konfirmasi_absen_gambar(self, user_id_terpilih, nama_terpilih, button_konfirmasi, face_crop):
        if not user_id_terpilih or user_id_terpilih == "Tidak Dikenal":
            self.status_label.configure(text="Pilihan tidak valid untuk absen.", text_color="orange")
            return

        # --- [PERBAIKAN BUG] Pastikan user ID valid sebelum akses DB ---
        if not isinstance(user_id_terpilih, int):
             try:
                  user_id_terpilih = int(user_id_terpilih)
             except ValueError:
                  self.status_label.configure(text="ID User terpilih tidak valid.", text_color="red")
                  return

        # Panggil mark_attendance DULU
        self.mark_attendance(nama_terpilih, user_id_terpilih, face_crop)
        
        # Lakukan Active Learning SETELAH absen dicatat
        try:
            # Periksa lagi apakah face_crop valid
            if face_crop is None or face_crop.size == 0:
                 print("WARNING: face_crop tidak valid saat Active Learning, update DB dibatalkan.")
                 raise ValueError("Data wajah tidak valid untuk embedding.")

            embedding_obj = DeepFace.represent(face_crop, model_name='SFace', enforce_detection=False, detector_backend='skip')
            new_embedding = embedding_obj[0]['embedding']
            
            # --- [PERBAIKAN BUG] Validasi user_id_terpilih sebelum akses DB ---
            if user_id_terpilih in self.embedding_db:
                 if 'embeddings' not in self.embedding_db[user_id_terpilih] or not isinstance(self.embedding_db[user_id_terpilih]['embeddings'], list):
                      old_embedding = self.embedding_db[user_id_terpilih].get('embedding')
                      self.embedding_db[user_id_terpilih]['embeddings'] = [old_embedding] if old_embedding else []
                      if 'embedding' in self.embedding_db[user_id_terpilih]:
                           del self.embedding_db[user_id_terpilih]['embedding']
                 
                 self.embedding_db[user_id_terpilih]['embeddings'].append(new_embedding)
                 
                 # --- [PERBAIKAN BUG] Update nama jika berbeda (jarang terjadi, tapi mungkin) ---
                 if self.embedding_db[user_id_terpilih]['name'] != nama_terpilih:
                      print(f"Mengupdate nama untuk ID {user_id_terpilih} dari '{self.embedding_db[user_id_terpilih]['name']}' ke '{nama_terpilih}'")
                      self.embedding_db[user_id_terpilih]['name'] = nama_terpilih

            else:
                # Ini seharusnya tidak terjadi jika dipilih dari combobox, tapi buat entri baru jika perlu
                print(f"Membuat entri baru untuk ID {user_id_terpilih} dengan nama {nama_terpilih}")
                self.embedding_db[user_id_terpilih] = {'name': nama_terpilih, 'embeddings': [new_embedding]}
            
            self.save_embedding_db()
            print(f"INFO: Active Learning Sukses! Database untuk '{nama_terpilih}' (ID: {user_id_terpilih}) telah diperbarui.")

        except ValueError as ve: # Tangkap error spesifik jika face_crop tidak valid
             print(f"ERROR saat Active Learning (ValueError): {ve}")
             self.status_label.configure(text=f"Absen tercatat, tapi gagal update DB: {ve}", text_color="orange")
        except Exception as e:
            print(f"ERROR saat Active Learning (Lainnya): {e}")
            self.status_label.configure(text=f"Absen tercatat, tapi gagal update DB: {e}", text_color="orange")

        button_konfirmasi.configure(text="Tercatat & Terupdate", state="disabled", fg_color="grey")


    def _recognize_one_face(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return "Crop Gagal", float('inf')
        try:
            live_embedding_obj = DeepFace.represent(face_crop, model_name='SFace', enforce_detection=False, detector_backend='skip')
            live_embedding = live_embedding_obj[0]['embedding']
            
            if not self.embedding_db: return "DB Kosong", float('inf')
            
            min_distance = float("inf")
            recognized_id = "Tidak Dikenal"

            # Salin item DB untuk iterasi yang aman
            db_items = list(self.embedding_db.items())

            for user_id, data in db_items:
                 # Cek format data
                 embeddings_to_check = []
                 if 'embeddings' in data and isinstance(data['embeddings'], list):
                      embeddings_to_check = data['embeddings']
                 elif 'embedding' in data: # Handle format lama
                      embeddings_to_check = [data['embedding']]
                 
                 if not embeddings_to_check: continue # Lewati jika tidak ada embedding valid

                 for db_embedding in embeddings_to_check:
                      # Pastikan embedding valid sebelum menghitung jarak
                      if db_embedding is not None and len(db_embedding) > 0:
                           try:
                                distance = cosine(live_embedding, db_embedding)
                                if distance < min_distance:
                                     min_distance = distance
                                     recognized_id = user_id
                           except ValueError as ve:
                                # Ini bisa terjadi jika dimensi embedding tidak cocok
                                print(f"Error cosine distance untuk ID {user_id}: {ve}. Ukuran live: {len(live_embedding)}, Ukuran DB: {len(db_embedding)}")
                                continue # Lanjut ke embedding berikutnya
                      else:
                           print(f"Peringatan: Embedding tidak valid ditemukan untuk ID {user_id}")


            # --- [PERBAIKAN BUG] Pastikan recognized_id benar-benar ada di DB ---
            if recognized_id != "Tidak Dikenal" and recognized_id not in self.embedding_db:
                 print(f"Peringatan: recognized_id {recognized_id} tidak ditemukan di DB setelah pencocokan. Direset ke Tidak Dikenal.")
                 recognized_id = "Tidak Dikenal"
                 min_distance = float('inf')

            return recognized_id, min_distance

        except ValueError as ve: # Menangkap error jika DeepFace gagal (misal wajah terlalu kecil)
             print(f"DEBUG: DeepFace represent error -> {ve}")
             return "Error Proses", float('inf')
        except Exception as e:
            print(f"DEBUG: Error saat mengenali wajah -> {e}")
            return "Error Proses", float('inf')


    # --- FUNGSI MANAJEMEN DATABASE ---
    def tampilkan_database(self):
        # ... (Kode Tampilkan Database sama seperti sebelumnya) ...
        if self.db_window is not None and self.db_window.winfo_exists():
            self.db_window.focus()
            return

        self.db_window = ctk.CTkToplevel(self)
        self.db_window.title("Database Wajah Terdaftar")
        self.db_window.geometry("500x600")
        self.db_window.transient(self)

        title_label = ctk.CTkLabel(self.db_window, text="Data Terdaftar", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10)

        scroll_frame = ctk.CTkScrollableFrame(self.db_window)
        scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.embedding_db = self.load_embedding_db()

        if not self.embedding_db:
            empty_label = ctk.CTkLabel(scroll_frame, text="Database masih kosong.", font=ctk.CTkFont(size=14))
            empty_label.pack(pady=20)
            return

        # Ambil item DB untuk iterasi aman
        db_items = list(self.embedding_db.items())

        for user_id, data in db_items:
            nama = data.get('name', 'N/A')
            # Cek format baru dan lama
            num_embeddings = len(data.get('embeddings', [])) if 'embeddings' in data else (1 if 'embedding' in data else 0)

            entry_frame = ctk.CTkFrame(scroll_frame, border_width=1, border_color="gray40")
            entry_frame.pack(pady=5, padx=5, fill="x")

            info_text = f"Nama: {nama}\nID Siswa: {user_id}\nJumlah Foto: {num_embeddings}"
            info_label = ctk.CTkLabel(entry_frame, text=info_text, justify="left")
            info_label.pack(side="left", padx=10, pady=10)

            delete_button = ctk.CTkButton(entry_frame, text="Hapus", fg_color="#D32F2F", hover_color="#B71C1C", width=80, command=lambda uid=user_id: self.hapus_user_dari_db(uid))
            delete_button.pack(side="right", padx=10, pady=10)

    def hapus_user_dari_db(self, user_id_to_delete):
        # Tambahkan konfirmasi sebelum menghapus
        # Pastikan parent diatur ke jendela DB agar messagebox muncul di atasnya
        parent_window = self.db_window if self.db_window and self.db_window.winfo_exists() else self
        confirm = messagebox.askyesno("Konfirmasi Hapus", f"Anda yakin ingin menghapus data untuk ID: {user_id_to_delete}?", parent=parent_window)
        if not confirm:
             return
             
        if user_id_to_delete in self.embedding_db:
            del self.embedding_db[user_id_to_delete]
            self.save_embedding_db()
            self.status_label.configure(text=f"User ID: {user_id_to_delete} berhasil dihapus.", text_color="lightgreen")
            
            # --- [PERBAIKAN BUG] Refresh jendela DB dengan benar ---
            if self.db_window is not None and self.db_window.winfo_exists():
                 # Hancurkan widget di dalam scroll_frame, bukan seluruh window
                 for widget in self.db_window.winfo_children():
                      # Hati-hati jangan hancurkan scroll_frame itu sendiri atau title
                      if isinstance(widget, ctk.CTkScrollableFrame) or isinstance(widget, ctk.CTkLabel) and widget != title_label:
                            widget.destroy() 
                 # Tutup dan buka lagi (lebih mudah untuk refresh total)
                 self.db_window.destroy()
                 self.tampilkan_database() 
            else:
                 # Jika jendela sudah ditutup, panggil saja lagi
                 self.tampilkan_database()
        else:
            self.status_label.configure(text=f"Gagal menghapus: User ID {user_id_to_delete} tidak ditemukan.", text_color="orange")
    
    # --- FUNGSI PENDAFTARAN WAJAH ---
    def unggah_gambar_daftar(self):
        filepath = filedialog.askopenfilename(title="Pilih Foto untuk Pendaftaran", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath: return
        frame = cv2.imread(filepath)
        if frame is None:
            self.status_label.configure(text="Gagal membaca file gambar.", text_color="red")
            return
        self.process_and_show_faces(frame)

    def process_and_show_faces(self, frame):
        self.batal_simpan_semua_wajah()
        results = face_detector_model(frame, verbose=False, conf=0.3)
        faces = results[0].boxes.xyxy.cpu().numpy().astype(int)

        if len(faces) > 0:
            self.status_label.configure(text=f"Ditemukan {len(faces)} wajah. Isi data di bawah.", text_color="cyan")
            self.preview_frame_container.pack(pady=10, padx=10, fill="both", expand=True)

            for (x1, y1, x2, y2) in faces:
                 # Pastikan koordinat valid sebelum crop
                 y1, y2 = max(0, y1), min(frame.shape[0], y2)
                 x1, x2 = max(0, x1), min(frame.shape[1], x2)
                 face_crop_color = frame[y1:y2, x1:x2]

                 if face_crop_color.size == 0: continue

                 try:
                      face_pil = Image.fromarray(cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2RGB))
                      ctk_image = ctk.CTkImage(light_image=face_pil, dark_image=face_pil, size=(100, 100))
                 except Exception as e:
                      print(f"Error membuat preview untuk pendaftaran: {e}")
                      ctk_image = None # Handle jika error

                 entry_frame = ctk.CTkFrame(self.preview_frame_container, border_width=1, border_color="gray30")
                 
                 # Tampilkan placeholder jika gambar gagal
                 if ctk_image:
                      preview_label = ctk.CTkLabel(entry_frame, image=ctk_image, text="")
                 else:
                      preview_label = ctk.CTkLabel(entry_frame, text="Gagal Load", width=100, height=100)
                 preview_label.pack(pady=10, padx=10)

                 name_entry = ctk.CTkEntry(entry_frame, placeholder_text="Nama Lengkap", width=180)
                 name_entry.pack(pady=(0, 5), padx=10)
                 id_entry = ctk.CTkEntry(entry_frame, placeholder_text="ID Siswa", width=180)
                 id_entry.pack(pady=5, padx=10)

                 # --- [PERBAIKAN BUG] Simpan face_crop ASLI, bukan PIL ---
                 face_data = {"face_image": face_crop_color, "name_widget": name_entry, "id_widget": id_entry}
                 self.detected_faces_data.append(face_data)
                
                 entry_frame.pack(side="top", pady=10, padx=10, fill="x")

            action_buttons_frame = ctk.CTkFrame(self.preview_frame_container, fg_color="transparent")
            save_all_button = ctk.CTkButton(action_buttons_frame, text="Simpan & Tambah ke Database", command=self.simpan_dan_buat_embedding)
            cancel_all_button = ctk.CTkButton(action_buttons_frame, text="Batal", command=self.batal_simpan_semua_wajah, fg_color="#D32F2F", hover_color="#B71C1C")
            action_buttons_frame.pack(pady=10)
            save_all_button.pack(side="left", padx=5)
            cancel_all_button.pack(side="right", padx=5)
        else:
            self.status_label.configure(text="Error: Tidak ada wajah yang terdeteksi.", text_color="orange")

    # --- [VERSI FINAL & PERBAIKAN BUG] ---
    def simpan_dan_buat_embedding(self):
        wajah_tersimpan = 0
        wajah_gagal_embedding = 0 # Tambah counter gagal
        for data in self.detected_faces_data:
            nama = data["name_widget"].get()
            student_id_str = data["id_widget"].get()
            print(f"Mencoba memvalidasi: Nama='{nama}', ID='{student_id_str}'") # Debug print
            
            if nama and student_id_str:
                try:
                    student_id = int(student_id_str.strip())
                    face_image = data["face_image"] # Ambil face_crop asli

                    # --- [PERBAIKAN BUG] Pastikan face_image valid ---
                    if face_image is None or face_image.size == 0:
                         print(f"Peringatan: face_image tidak valid untuk ID {student_id_str}. Melewati.")
                         wajah_gagal_embedding += 1
                         continue

                    try:
                        embedding_obj = DeepFace.represent(img_path=face_image, model_name='SFace', enforce_detection=False)
                        if not embedding_obj or not embedding_obj[0].get('embedding'):
                             print(f"Peringatan: DeepFace tidak mengembalikan embedding untuk ID {student_id}. Melewati.")
                             wajah_gagal_embedding += 1
                             continue

                        embedding = embedding_obj[0]['embedding']

                        if student_id in self.embedding_db:
                            if 'embeddings' not in self.embedding_db[student_id] or not isinstance(self.embedding_db[student_id]['embeddings'], list):
                                old_embedding = self.embedding_db[student_id].get('embedding')
                                self.embedding_db[student_id]['embeddings'] = [old_embedding] if old_embedding else []
                                if 'embedding' in self.embedding_db[student_id]:
                                    del self.embedding_db[student_id]['embedding']
                                    
                            self.embedding_db[student_id]['embeddings'].append(embedding)
                        else:
                            self.embedding_db[student_id] = {'name': nama, 'embeddings': [embedding]}
                        wajah_tersimpan += 1
                        print(f"Berhasil memproses & menambahkan embedding untuk ID: {student_id}") # Debug print
                    
                    except ValueError as ve: # Tangkap error spesifik dari DeepFace jika wajah terlalu kecil dll
                        print(f"DEBUG: DeepFace represent error untuk ID {student_id}: {ve}")
                        self.status_label.configure(text=f"Gagal embedding wajah ID {student_id}: Wajah mungkin terlalu kecil/tidak jelas.", text_color="orange")
                        wajah_gagal_embedding += 1
                        continue
                    except Exception as e:
                        print(f"DEBUG: DeepFace Error (Lainnya) -> {e}")
                        self.status_label.configure(text=f"Error saat membuat embedding: {e}", text_color="red")
                        wajah_gagal_embedding += 1
                        continue
                except ValueError:
                    print(f"ID tidak valid dilewati: '{student_id_str}'")
                    self.status_label.configure(text=f"ID '{student_id_str}' harus berupa angka.", text_color="orange")
                    continue
        
        # Berikan status akhir yang lebih informatif
        if wajah_tersimpan > 0:
            self.save_embedding_db()
            status_akhir = f"Sukses! {wajah_tersimpan} data ditambahkan."
            if wajah_gagal_embedding > 0:
                 status_akhir += f" ({wajah_gagal_embedding} wajah gagal di-embed)."
            self.status_label.configure(text=status_akhir, text_color="lightgreen")
        else:
            status_akhir = "Tidak ada data valid yang berhasil disimpan."
            if wajah_gagal_embedding > 0:
                 status_akhir += f" ({wajah_gagal_embedding} wajah gagal di-embed)."
            else:
                 status_akhir += " Pastikan Nama & ID diisi benar."
            self.status_label.configure(text=status_akhir, text_color="orange")
        
        self.batal_simpan_semua_wajah()


    # --- FUNGSI HELPER & DATABASE ---
    def load_embedding_db(self):
        # ... (Kode load DB sama seperti sebelumnya) ...
        if os.path.exists(embedding_db_path):
            try:
                with open(embedding_db_path, 'rb') as f:
                    # --- [PENTING] Tambahkan penanganan format file yang mungkin korup ---
                    data = pickle.load(f)
                    # Lakukan validasi dasar pada data yang dimuat
                    if isinstance(data, dict):
                         # Periksa format internal (opsional tapi bagus)
                         for key, value in data.items():
                              if not isinstance(key, int) or not isinstance(value, dict) or \
                                 ('name' not in value or not isinstance(value['name'], str)) or \
                                 ('embeddings' not in value or not isinstance(value['embeddings'], list)):
                                  print(f"Peringatan: Format data tidak valid ditemukan di DB untuk key {key}. Mungkin perlu dihapus manual.")
                                  # Anda bisa menambahkan logika untuk memperbaiki atau menghapus data buruk di sini
                         return data
                    else:
                         print("ERROR: File embeddings.pkl tidak berisi dictionary. Membuat DB baru.")
                         return {}
            except (pickle.UnpicklingError, EOFError):
                print("WARNING: File embeddings.pkl rusak atau kosong. Membuat database baru.")
                return {}
            except Exception as e:
                print(f"ERROR saat memuat DB: {e}")
                return {}
        print("File embeddings.pkl tidak ditemukan. Membuat database baru.")
        return {}


    def save_embedding_db(self):
         # ... (Kode save DB sama seperti sebelumnya) ...
         try:
              with open(embedding_db_path, 'wb') as f:
                   pickle.dump(self.embedding_db, f)
              print("Database embedding berhasil disimpan.") # Debug print
         except Exception as e:
              print(f"ERROR saat menyimpan DB: {e}")
              self.status_label.configure(text=f"Gagal menyimpan database: {e}", text_color="red")


    def batal_simpan_semua_wajah(self):
        # ... (Kode batal simpan sama seperti sebelumnya) ...
        for widget in self.preview_frame_container.winfo_children():
            widget.destroy()
        self.preview_frame_container.pack_forget()
        self.detected_faces_data.clear()

    def mark_attendance(self, name, user_id, face_crop):
        # --- [FIX] Pindahkan pengecekan thread ke sini ---
        try:
            log_photo_dir = os.path.join(APP_PATH, "log_absensi", "foto_log")
            os.makedirs(log_photo_dir, exist_ok=True)
            now = datetime.now()
            timestamp_str = now.strftime('%Y%m%d_%H%M%S') 
            filename = f"{user_id}_{timestamp_str}.jpg"
            photo_path = os.path.join(log_photo_dir, filename)
            
            # Periksa apakah face_crop valid sebelum menyimpan
            if face_crop is not None and face_crop.size > 0:
                 # Coba simpan, tangani error jika gagal
                 try:
                      save_success = cv2.imwrite(photo_path, face_crop)
                      if not save_success:
                           raise IOError(f"imwrite gagal menyimpan ke {photo_path}")
                 except Exception as img_e:
                      print(f"ERROR: Gagal menyimpan foto bukti untuk ID {user_id}: {img_e}")
                      photo_path = "N/A (Gagal simpan foto)"
            else:
                 photo_path = "N/A (Data wajah tidak valid)"
                 print(f"WARNING: face_crop tidak valid untuk {name} ({user_id}), tidak menyimpan foto.")

            file_path_csv = os.path.join(APP_PATH, "log_absensi", "Attendance.csv")
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H:%M:%S')
            file_exists = os.path.isfile(file_path_csv)
            
            with open(file_path_csv, 'a', newline='', encoding='utf-8') as f:
                if not file_exists:
                    f.write("Nama;ID Siswa;Tanggal;Waktu;Path Foto\n")
                f.write(f"{name};{user_id};{date_str};{time_str};{photo_path}\n")
            
            # --- [PERBAIKAN] Gunakan self.after untuk update GUI dari thread ---
            # Cara ini lebih aman daripada cek _get_running_app
            self.after(0, lambda: self.status_label.configure(text=f"Absen Dicatat: {name} | {time_str}", text_color="lightgreen"))
            
            print(f"Absen Dicatat: {name} | {time_str}") # Print ke konsol selalu aman

        except Exception as e:
            error_msg = f"Gagal menyimpan absen/foto: {e}"
            # Gunakan self.after juga untuk update error
            self.after(0, lambda: self.status_label.configure(text=error_msg, text_color="red"))
            print(f"ERROR: {error_msg}")


if __name__ == "__main__":
    app = SmartAttendanceApp()
    app.mainloop()

