import customtkinter as ctk
import cv2
import os
import sys
import numpy as np
from PIL import Image
from tkinter import filedialog
from datetime import datetime
import pickle
from ultralytics import YOLO
from deepface import DeepFace
from scipy.spatial.distance import cosine

# --- [FUNGSI KUNCI] Menentukan Path Aplikasi ---
def get_app_path():
    """Mendapatkan path root aplikasi, berfungsi untuk mode dev dan .exe."""
    if getattr(sys, 'frozen', False):
        # Jika dijalankan sebagai .exe (dibekukan oleh PyInstaller)
        return os.path.dirname(sys.executable)
    else:
        # Jika dijalankan sebagai skrip .py biasa
        return os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path):
    """ Mendapatkan path absolut ke file sumber daya (seperti model .pt)."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

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
        self.detected_faces_data = []
        self.embedding_db = self.load_embedding_db()
        self.db_window = None # Untuk melacak jendela manajemen DB

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

        # --- BAGIAN 1: PENGAMBILAN DATA WAJAH ---
        self.input_method_label = ctk.CTkLabel(self.master_scroll_frame, text="1. Tambah Data Wajah",
                                               font=ctk.CTkFont(size=16, weight="bold"))
        self.input_method_label.pack(pady=(5, 5), anchor="w", padx=10)
        self.tambah_data_button = ctk.CTkButton(self.master_scroll_frame, text="Unggah Foto untuk Pendaftaran",
                                                height=40, command=self.unggah_gambar)
        self.tambah_data_button.pack(pady=5, fill="x", padx=10)

        self.preview_frame_container = ctk.CTkFrame(self.master_scroll_frame, fg_color="transparent")

        # --- BAGIAN 2: ABSENSI ---
        self.attend_label = ctk.CTkLabel(self.master_scroll_frame, text="2. Mulai Absensi",
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.attend_label.pack(pady=(20, 5), anchor="w", padx=10)
        self.start_attendance_button = ctk.CTkButton(self.master_scroll_frame, text="Unggah & Proses Foto Kelas",
                                                     height=40, command=self.absensiDariGambar)
        self.start_attendance_button.pack(pady=5, fill="x", padx=10)

        # --- BAGIAN 3: MANAJEMEN DATABASE ---
        self.manage_label = ctk.CTkLabel(self.master_scroll_frame, text="3. Manajemen Database",
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.manage_label.pack(pady=(20, 5), anchor="w", padx=10)
        self.view_db_button = ctk.CTkButton(self.master_scroll_frame, text="Lihat & Kelola Database", height=40,
                                              fg_color="#1F6AA5", hover_color="#144E7A",
                                              command=self.tampilkan_database)
        self.view_db_button.pack(pady=5, fill="x", padx=10)

    # --- FUNGSI ABSENSI DARI GAMBAR UNTUK MULTI-WAJAH ---
    def absensiDariGambar(self):
        filepath = filedialog.askopenfilename(
            title="Pilih Gambar untuk Absensi",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath: return

        frame = cv2.imread(filepath)
        if frame is None:
            self.status_label.configure(text="Gagal membaca file gambar.", text_color="red")
            return

        self.status_label.configure(text="Memproses gambar, mohon tunggu...", text_color="cyan")
        self.update_idletasks()

        THRESHOLD_YAKIN_GAMBAR = 0.45

        results = face_detector_model(frame, verbose=False, conf=0.4)
        all_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        if len(all_boxes) == 0:
            self.status_label.configure(text="Tidak ada wajah terdeteksi.", text_color="orange")
            return

        konfirmasi_window = ctk.CTkToplevel(self)
        konfirmasi_window.title("Menu Review Absensi")
        konfirmasi_window.geometry("600x700")
        konfirmasi_window.transient(self)

        scroll_frame = ctk.CTkScrollableFrame(konfirmasi_window, label_text="Hasil Deteksi - Konfirmasi Absensi di Bawah")
        scroll_frame.pack(fill="both", expand=True, padx=(10,50), pady=10)

        pilihan_nama = {"Tidak Dikenal": "Tidak Dikenal"}
        for user_id, data in self.embedding_db.items():
            pilihan_nama[data['name']] = user_id

        absen_otomatis_count = 0

        for i, box in enumerate(all_boxes):
            x1, y1, x2, y2 = box
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0: continue

            recognized_id, min_distance = self._kenali_satu_wajah(face_crop)
            
            card_frame = ctk.CTkFrame(scroll_frame, border_width=1)
            card_frame.pack(fill="x", padx=10, pady=10)

            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            ctk_image = ctk.CTkImage(light_image=face_pil, dark_image=face_pil, size=(100, 100))
            preview_label = ctk.CTkLabel(card_frame, image=ctk_image, text="")
            preview_label.pack(side="left", padx=10, pady=5)

            info_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
            info_frame.pack(side="left", fill="x", expand=True, padx=10)
            
            is_confident = False
            nama_dikenali = "N/A"
            if recognized_id not in ["Tidak Dikenal", "Crop Gagal", "Error Proses", "DB Kosong"]:
                if min_distance < THRESHOLD_YAKIN_GAMBAR:
                    is_confident = True
                nama_dikenali = self.embedding_db.get(recognized_id, {}).get('name', 'Error ID')

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
                if recognized_id not in ["Tidak Dikenal", "Crop Gagal", "Error Proses", "DB Kosong"]:
                    nama_saran = nama_dikenali
                    keyakinan = (1 - min_distance) * 100
                    saran_sistem = f"Saran: {nama_saran} (Yakin: {keyakinan:.0f}%)"
            
                saran_label = ctk.CTkLabel(info_frame, text=saran_sistem, font=ctk.CTkFont(size=12))
                saran_label.pack(anchor="w", pady=(5,0))

                nama_combobox = ctk.CTkComboBox(info_frame, values=list(pilihan_nama.keys()), width=200)
                nama_combobox.set(nama_saran)
                nama_combobox.pack(anchor="w", pady=5)

                konfirmasi_button = ctk.CTkButton(info_frame, text="Konfirmasi & Catat Absen")
                konfirmasi_button.pack(anchor="w", pady=10)
                konfirmasi_button.configure(command=lambda nc=nama_combobox, pn=pilihan_nama, btn=konfirmasi_button, fc=face_crop: 
                    self._handle_konfirmasi_absen_gambar(pn.get(nc.get()), nc.get(), btn, fc))
        
        status_text_akhir = f"Selesai! {absen_otomatis_count} absen otomatis, sisanya butuh konfirmasi."
        self.status_label.configure(text=status_text_akhir, text_color="white")

    def _handle_konfirmasi_absen_gambar(self, user_id_terpilih, nama_terpilih, button_konfirmasi, face_crop):
        if not user_id_terpilih or user_id_terpilih == "Tidak Dikenal":
            self.status_label.configure(text="Pilihan tidak valid untuk absen.", text_color="orange")
            return

        self.mark_attendance(nama_terpilih, user_id_terpilih, face_crop)
        
        try:
            embedding_obj = DeepFace.represent(face_crop, model_name='SFace', enforce_detection=False, detector_backend='skip')
            new_embedding = embedding_obj[0]['embedding']
            
            if user_id_terpilih in self.embedding_db:
                self.embedding_db[user_id_terpilih]['embeddings'].append(new_embedding)
            else:
                self.embedding_db[user_id_terpilih] = {'name': nama_terpilih, 'embeddings': [new_embedding]}
            
            self.save_embedding_db()
            print(f"INFO: Active Learning Sukses! Database untuk '{nama_terpilih}' telah diperbarui.")
        except Exception as e:
            print(f"ERROR saat Active Learning: {e}")
            self.status_label.configure(text=f"Absen tercatat, tapi gagal update DB: {e}", text_color="orange")

        button_konfirmasi.configure(text="Tercatat & Terupdate", state="disabled", fg_color="grey")

    def _kenali_satu_wajah(self, face_crop):
        if face_crop.size == 0:
            return "Crop Gagal", float('inf')
        try:
            live_embedding_obj = DeepFace.represent(face_crop, model_name='SFace', enforce_detection=False, detector_backend='skip')
            live_embedding = live_embedding_obj[0]['embedding']
            
            if not self.embedding_db: return "DB Kosong", float('inf')
            
            min_distance = float("inf")
            recognized_id = "Tidak Dikenal"

            for user_id, data in self.embedding_db.items():
                if 'embeddings' in data and data['embeddings']:
                    for db_embedding in data['embeddings']:
                        distance = cosine(live_embedding, db_embedding)
                        if distance < min_distance:
                            min_distance = distance
                            recognized_id = user_id
            return recognized_id, min_distance
        except Exception as e:
            print(f"DEBUG: Error saat mengenali wajah -> {e}")
            return "Error Proses", float('inf')

    def tampilkan_database(self):
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

        for user_id, data in self.embedding_db.items():
            nama = data.get('name', 'N/A')
            num_embeddings = len(data.get('embeddings', []))

            entry_frame = ctk.CTkFrame(scroll_frame, border_width=1, border_color="gray40")
            entry_frame.pack(pady=5, padx=5, fill="x")

            info_text = f"Nama: {nama}\nID Siswa: {user_id}\nJumlah Foto: {num_embeddings}"
            info_label = ctk.CTkLabel(entry_frame, text=info_text, justify="left")
            info_label.pack(side="left", padx=10, pady=10)

            delete_button = ctk.CTkButton(entry_frame, text="Hapus", fg_color="#D32F2F", hover_color="#B71C1C", width=80, command=lambda uid=user_id: self.hapus_user_dari_db(uid))
            delete_button.pack(side="right", padx=10, pady=10)

    def hapus_user_dari_db(self, user_id_to_delete):
        if user_id_to_delete in self.embedding_db:
            del self.embedding_db[user_id_to_delete]
            self.save_embedding_db()
            self.status_label.configure(text=f"User ID: {user_id_to_delete} berhasil dihapus.", text_color="lightgreen")
            
            if self.db_window is not None and self.db_window.winfo_exists():
                self.db_window.destroy()
            self.tampilkan_database()
        else:
            self.status_label.configure(text=f"Gagal menghapus: User ID {user_id_to_delete} tidak ditemukan.", text_color="orange")
    
    def unggah_gambar(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath: return
        frame = cv2.imread(filepath)
        if frame is None:
            self.status_label.configure(text="Gagal membaca file gambar.", text_color="red")
            return
        self.proses_dan_tampilan_wajah(frame)

    def proses_dan_tampilan_wajah(self, frame):
        self.batal_simpan_semua_wajah()
        results = face_detector_model(frame, verbose=False, conf=0.3)
        faces = results[0].boxes.xyxy.cpu().numpy().astype(int)

        if len(faces) > 0:
            self.status_label.configure(text=f"Ditemukan {len(faces)} wajah. Isi data di bawah.", text_color="cyan")
            self.preview_frame_container.pack(pady=10, padx=10, fill="both", expand=True)

            for (x1, y1, x2, y2) in faces:
                face_crop_color = frame[y1:y2, x1:x2]
                if face_crop_color.size == 0: continue

                face_pil = Image.fromarray(cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2RGB))
                ctk_image = ctk.CTkImage(light_image=face_pil, dark_image=face_pil, size=(100, 100))

                entry_frame = ctk.CTkFrame(self.preview_frame_container, border_width=1, border_color="gray30")
                preview_label = ctk.CTkLabel(entry_frame, image=ctk_image, text="")
                preview_label.pack(pady=10, padx=10)
                name_entry = ctk.CTkEntry(entry_frame, placeholder_text="Nama Lengkap", width=180)
                name_entry.pack(pady=(0, 5), padx=10)
                id_entry = ctk.CTkEntry(entry_frame, placeholder_text="ID Siswa", width=180)
                id_entry.pack(pady=5, padx=10)

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

    def simpan_dan_buat_embedding(self):
        wajah_tersimpan = 0
        for data in self.detected_faces_data:
            nama = data["name_widget"].get()
            student_id_str = data["id_widget"].get()
            
            if nama and student_id_str:
                try:
                    student_id = int(student_id_str.strip())
                    face_image = data["face_image"]

                    try:
                        embedding_obj = DeepFace.represent(img_path=face_image, model_name='SFace', enforce_detection=False)
                        embedding = embedding_obj[0]['embedding']

                        if student_id in self.embedding_db:
                            if 'embeddings' in self.embedding_db[student_id]:
                                self.embedding_db[student_id]['embeddings'].append(embedding)
                            else:
                                old_embedding = self.embedding_db[student_id]['embedding']
                                self.embedding_db[student_id]['embeddings'] = [old_embedding, embedding]
                                del self.embedding_db[student_id]['embedding']
                        else:
                            self.embedding_db[student_id] = {'name': nama, 'embeddings': [embedding]}
                        wajah_tersimpan += 1
                    except Exception as e:
                        print(f"DEBUG: DeepFace Error -> {e}")
                        self.status_label.configure(text=f"Error saat membuat embedding: {e}", text_color="red")
                        continue
                except ValueError:
                    print(f"ID tidak valid dilewati: '{student_id_str}'")
                    self.status_label.configure(text=f"ID '{student_id_str}' harus berupa angka.", text_color="orange")
                    continue
        
        if wajah_tersimpan > 0:
            self.save_embedding_db()
            self.status_label.configure(text=f"Sukses! {wajah_tersimpan} data ditambahkan.", text_color="lightgreen")
        else:
            self.status_label.configure(text="Tidak ada data valid untuk disimpan. Pastikan Nama & ID diisi benar.", text_color="orange")
        
        self.batal_simpan_semua_wajah()

    def load_embedding_db(self):
        if os.path.exists(embedding_db_path):
            try:
                with open(embedding_db_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                return {}
        return {}

    def save_embedding_db(self):
        with open(embedding_db_path, 'wb') as f:
            pickle.dump(self.embedding_db, f)

    def batal_simpan_semua_wajah(self):
        for widget in self.preview_frame_container.winfo_children():
            widget.destroy()
        self.preview_frame_container.pack_forget()
        self.detected_faces_data.clear()

    def mark_attendance(self, name, user_id, face_crop):
        try:
            log_photo_dir = os.path.join(APP_PATH, "log_absensi", "foto_log")
            os.makedirs(log_photo_dir, exist_ok=True)
            now = datetime.now()
            timestamp_str = now.strftime('%Y%m%d_%H%M%S') 
            filename = f"{user_id}_{timestamp_str}.jpg"
            photo_path = os.path.join(log_photo_dir, filename)
            cv2.imwrite(photo_path, face_crop)

            file_path_csv = os.path.join(APP_PATH, "log_absensi", "Attendance.csv")
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H:%M:%S')
            file_exists = os.path.isfile(file_path_csv)
            
            with open(file_path_csv, 'a', newline='', encoding='utf-8') as f:
                if not file_exists:
                    f.write("Nama;ID Siswa;Tanggal;Waktu;Path Foto\n")
                f.write(f"{name};{user_id};{date_str};{time_str};{photo_path}\n")
            
            self.status_label.configure(text=f"Absen Dicatat: {name} | {time_str}", text_color="lightgreen")
        except Exception as e:
            self.status_label.configure(text=f"Gagal menyimpan absen/foto: {e}", text_color="red")
            print(f"Error saat menulis absen/foto: {e}")

if __name__ == "__main__":
    app = SmartAttendanceApp()
    app.mainloop()
