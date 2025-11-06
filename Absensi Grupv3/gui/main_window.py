import customtkinter as ctk
import cv2
import os
import time
from PIL import Image
from tkinter import filedialog, messagebox
from datetime import datetime

# Import core components
from core.face_detection import FaceDetector
from core.face_recognition import FaceRecognizer
from core.database_manager import DatabaseManager
from core.attendance_logger import AttendanceLogger
from core.utils import get_app_path
from gui.database_window import DatabaseWindow
from gui.review_window import ReviewWindow

class SmartAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.db_manager = DatabaseManager()
        self.attendance_logger = AttendanceLogger()
        
        # GUI variables
        self.camera_index = 0
        self.detected_faces_data = []
        self.db_window = None
        self.timer_job_id = None
        self.current_repetition = 0
        self.total_repetitions = 0
        self.timer_interval_ms = 0
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup antarmuka utama"""
        # --- KONFIGURASI JENDELA & TEMA ---
        self.title("Smart Attendance [YOLO + SFace]")
        self.geometry("450x700")
        self.resizable(True, True)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

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

        # --- Pilihan Input (Checkbox) ---
        self.setup_input_section()
        
        # --- Bagian 1: Pengambilan Data Wajah ---
        self.setup_face_registration_section()
        
        # --- Bagian 2: Absensi ---
        self.setup_attendance_section()
        
        # --- Bagian 3: Manajemen Database ---
        self.setup_database_section()

        # Panggil toggle mode di akhir untuk setup awal UI
        self._toggle_input_mode()

    def setup_input_section(self):
        """Setup section pilihan input"""
        input_option_frame = ctk.CTkFrame(self.master_scroll_frame, fg_color="transparent")
        input_option_frame.pack(fill="x", padx=10, pady=5)
        
        self.use_camera_var = ctk.BooleanVar(value=False)
        self.use_camera_checkbox = ctk.CTkCheckBox(
            input_option_frame, 
            text="Gunakan Kamera (untuk Daftar & Absen Timer)",
            variable=self.use_camera_var, 
            font=ctk.CTkFont(size=14),
            command=self._toggle_input_mode
        )
        self.use_camera_checkbox.pack(pady=(5, 10), anchor="w")

    def setup_face_registration_section(self):
        """Setup section pendaftaran wajah"""
        self.input_method_label = ctk.CTkLabel(
            self.master_scroll_frame, 
            text="1. Tambah Data Wajah",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.input_method_label.pack(pady=(5, 5), anchor="w", padx=10)
        
        self.tambah_data_button = ctk.CTkButton(
            self.master_scroll_frame, 
            text="Unggah Foto Pendaftaran",
            height=40, 
            command=self.handle_tambah_data
        )
        self.tambah_data_button.pack(pady=5, fill="x", padx=10)

        self.preview_frame_container = ctk.CTkFrame(self.master_scroll_frame, fg_color="transparent")

    def setup_attendance_section(self):
        """Setup section absensi"""
        self.attend_label = ctk.CTkLabel(
            self.master_scroll_frame, 
            text="2. Mulai Absensi",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.attend_label.pack(pady=(20, 5), anchor="w", padx=10)
        
        opsi_absen_frame = ctk.CTkFrame(self.master_scroll_frame, fg_color="transparent")
        opsi_absen_frame.pack(fill="x", padx=10, pady=5)
        
        # Tombol Absen via Unggah Foto
        self.upload_absen_button = ctk.CTkButton(
            opsi_absen_frame, 
            text="Absen via Unggah Foto", 
            height=40,
            command=self.absensi_dari_gambar
        )
        self.upload_absen_button.pack(fill="x", pady=(0, 10))

        # Opsi Timer
        self.setup_timer_section(opsi_absen_frame)

    def setup_timer_section(self, parent_frame):
        """Setup section timer absensi"""
        self.timer_frame = ctk.CTkFrame(parent_frame)
        
        timer_label = ctk.CTkLabel(
            self.timer_frame, 
            text="Absensi Otomatis via Timer (Gunakan Kamera):"
        )
        timer_label.pack(anchor="w", padx=10, pady=(5,0))

        input_frame = ctk.CTkFrame(self.timer_frame, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        # Interval input
        interval_label = ctk.CTkLabel(input_frame, text="Interval (menit):")
        interval_label.pack(side="left", padx=(0, 5))
        self.interval_entry = ctk.CTkEntry(input_frame, width=60, placeholder_text="10")
        self.interval_entry.pack(side="left", padx=5)
        self.interval_entry.insert(0, "10")

        # Repeat input
        repeat_label = ctk.CTkLabel(input_frame, text="Ulangi (kali):")
        repeat_label.pack(side="left", padx=(10, 5))
        self.repeat_entry = ctk.CTkEntry(input_frame, width=60, placeholder_text="2")
        self.repeat_entry.pack(side="left", padx=5)
        self.repeat_entry.insert(0, "2")

        # Camera selection
        self.camera_combobox = ctk.CTkComboBox(
            self.timer_frame, 
            height=35, 
            command=self.set_camera_index
        )
        self.camera_combobox.pack(pady=(10, 5), fill="x", padx=10)
        self.detect_cameras()

        # Start timer button
        self.start_timer_button = ctk.CTkButton(
            self.timer_frame, 
            text="Mulai Absen Terjadwal Kamera", 
            height=35,
            command=self.mulai_absensi_terjadwal_kamera
        )
        self.start_timer_button.pack(fill="x", padx=10, pady=(5, 10))

    def setup_database_section(self):
        """Setup section manajemen database"""
        self.manage_label = ctk.CTkLabel(
            self.master_scroll_frame, 
            text="3. Manajemen Database",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.manage_label.pack(pady=(20, 5), anchor="w", padx=10)
        
        self.view_db_button = ctk.CTkButton(
            self.master_scroll_frame, 
            text="Lihat & Kelola Database", 
            height=40,
            fg_color="#1F6AA5", 
            hover_color="#144E7A",
            command=self.tampilkan_database
        )
        self.view_db_button.pack(pady=5, fill="x", padx=10)

    # --- INPUT MODE TOGGLE ---
    def _toggle_input_mode(self):
        """Toggle antara mode kamera dan upload file"""
        use_camera = self.use_camera_var.get()
        if use_camera:
            self.tambah_data_button.configure(
                text="Ambil Foto Pendaftaran via Kamera", 
                command=self.ambil_dari_kamera
            )
            self.timer_frame.pack(fill="x", pady=5)
            self.detect_cameras()
            if self.camera_index == -1:
                self.start_timer_button.configure(state="disabled")
            else:
                self.start_timer_button.configure(state="normal")
        else:
            self.tambah_data_button.configure(
                text="Unggah Foto Pendaftaran", 
                command=self.unggah_gambar_daftar
            )
            self.timer_frame.pack_forget()

    def handle_tambah_data(self):
        """Dispatcher untuk tambah data berdasarkan mode input"""
        current_command = self.tambah_data_button.cget("command")
        if current_command:
            current_command()

    # --- CAMERA FUNCTIONS ---
    def detect_cameras(self):
        """Mendeteksi kamera yang tersedia"""
        available_cameras = []
        try:
            for i in range(5):
                cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
                if cap.isOpened():
                    available_cameras.append(f"Kamera {i}")
                    cap.release()
                else:
                    break
        except Exception as e:
            print(f"Error saat deteksi kamera: {e}")

        if available_cameras:
            self.camera_combobox.configure(values=available_cameras, state="normal")
            current_selection = f"Kamera {self.camera_index}"
            if current_selection in available_cameras:
                self.camera_combobox.set(current_selection)
            else:
                self.camera_combobox.set(available_cameras[0])
                self.camera_index = 0
            if self.use_camera_var.get():
                self.start_timer_button.configure(state="normal")
        else:
            self.camera_combobox.configure(values=["Tidak Ada Kamera"], state="disabled")
            self.camera_combobox.set("Tidak Ada Kamera")
            self.camera_index = -1
            self.status_label.configure(text="Tidak ada kamera terdeteksi!", text_color="orange")
            self.start_timer_button.configure(state="disabled")

    def set_camera_index(self, choice):
        """Set indeks kamera yang dipilih"""
        try:
            if "Kamera" in choice:
                self.camera_index = int(choice.split(' ')[1])
                self.status_label.configure(text=f"{choice} terpilih.", text_color="white")
                if self.use_camera_var.get():
                    self.start_timer_button.configure(state="normal")
            else:
                self.camera_index = -1
                self.start_timer_button.configure(state="disabled")
        except (ValueError, IndexError):
            self.camera_index = -1
            self.start_timer_button.configure(state="disabled")

    # --- FACE REGISTRATION FUNCTIONS ---
    def unggah_gambar_daftar(self):
        """Unggah gambar untuk pendaftaran wajah"""
        filepath = filedialog.askopenfilename(
            title="Pilih Foto untuk Pendaftaran", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath: 
            return
            
        frame = cv2.imread(filepath)
        if frame is None:
            self.status_label.configure(text="Gagal membaca file gambar.", text_color="red")
            return
            
        self.process_and_show_faces(frame)

    def ambil_dari_kamera(self):
        """Ambil foto dari kamera untuk pendaftaran"""
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
            if not ret: 
                break

            # Deteksi wajah untuk visual feedback
            results_viz = self.face_detector.detect_faces_visual(frame)
            frame_disp = frame.copy()
            
            for r in results_viz:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow(window_name, frame_disp)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.process_and_show_faces(frame)
                break
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.detect_cameras()

    def process_and_show_faces(self, frame):
        """Process detected faces and show in GUI"""
        # Clear previous faces
        self.batal_simpan_semua_wajah()
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame, conf=0.3)

        if len(faces) > 0:
            self.status_label.configure(
                text=f"Ditemukan {len(faces)} wajah. Isi data di bawah.", 
                text_color="cyan"
            )
            self.preview_frame_container.pack(pady=10, padx=10, fill="both", expand=True)

            for (x1, y1, x2, y2) in faces:
                # Ensure valid coordinates
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                face_crop_color = frame[y1:y2, x1:x2]

                if face_crop_color.size == 0: 
                    continue

                # Create preview
                try:
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2RGB))
                    ctk_image = ctk.CTkImage(light_image=face_pil, dark_image=face_pil, size=(100, 100))
                except Exception as e:
                    print(f"Error membuat preview: {e}")
                    ctk_image = None

                # Create entry frame for each face
                self.create_face_entry_frame(face_crop_color, ctk_image)

            # Add action buttons
            self.create_action_buttons()
        else:
            self.status_label.configure(text="Error: Tidak ada wajah yang terdeteksi.", text_color="orange")

    def create_face_entry_frame(self, face_crop, ctk_image):
        """Create entry frame for each detected face"""
        entry_frame = ctk.CTkFrame(self.preview_frame_container, border_width=1, border_color="gray30")
        
        # Preview image
        if ctk_image:
            preview_label = ctk.CTkLabel(entry_frame, image=ctk_image, text="")
        else:
            preview_label = ctk.CTkLabel(entry_frame, text="Gagal Load", width=100, height=100)
        preview_label.pack(pady=10, padx=10)

        # Name entry
        name_entry = ctk.CTkEntry(entry_frame, placeholder_text="Nama Lengkap", width=180)
        name_entry.pack(pady=(0, 5), padx=10)
        
        # ID entry
        id_entry = ctk.CTkEntry(entry_frame, placeholder_text="ID Siswa", width=180)
        id_entry.pack(pady=5, padx=10)

        # Store face data
        face_data = {
            "face_image": face_crop, 
            "name_widget": name_entry, 
            "id_widget": id_entry
        }
        self.detected_faces_data.append(face_data)
        
        entry_frame.pack(side="top", pady=10, padx=10, fill="x")

    def create_action_buttons(self):
        """Create save and cancel buttons"""
        action_buttons_frame = ctk.CTkFrame(self.preview_frame_container, fg_color="transparent")
        
        save_all_button = ctk.CTkButton(
            action_buttons_frame, 
            text="Simpan & Tambah ke Database", 
            command=self.simpan_dan_buat_embedding
        )
        
        cancel_all_button = ctk.CTkButton(
            action_buttons_frame, 
            text="Batal", 
            command=self.batal_simpan_semua_wajah, 
            fg_color="#D32F2F", 
            hover_color="#B71C1C"
        )
        
        action_buttons_frame.pack(pady=10)
        save_all_button.pack(side="left", padx=5)
        cancel_all_button.pack(side="right", padx=5)

    def simpan_dan_buat_embedding(self):
        """Save faces to database with embeddings"""
        wajah_tersimpan = 0
        wajah_gagal_embedding = 0
        
        for data in self.detected_faces_data:
            nama = data["name_widget"].get()
            student_id_str = data["id_widget"].get()
            
            if nama and student_id_str:
                try:
                    student_id = int(student_id_str.strip())
                    face_image = data["face_image"]

                    if face_image is None or face_image.size == 0:
                        wajah_gagal_embedding += 1
                        continue

                    # Extract embedding
                    embedding = self.face_recognizer.extract_embedding(face_image)
                    if embedding is None:
                        wajah_gagal_embedding += 1
                        continue

                    # Add to database
                    if self.db_manager.add_user(student_id, nama, embedding):
                        wajah_tersimpan += 1

                except ValueError:
                    self.status_label.configure(
                        text=f"ID '{student_id_str}' harus berupa angka.", 
                        text_color="orange"
                    )
                    continue
        
        # Show result
        if wajah_tersimpan > 0:
            self.db_manager.save_database()
            status_akhir = f"Sukses! {wajah_tersimpan} data ditambahkan."
            if wajah_gagal_embedding > 0:
                status_akhir += f" ({wajah_gagal_embedding} wajah gagal di-embed)."
            self.status_label.configure(text=status_akhir, text_color="lightgreen")
        else:
            status_akhir = "Tidak ada data valid yang berhasil disimpan."
            if wajah_gagal_embedding > 0:
                status_akhir += f" ({wajah_gagal_embedding} wajah gagal di-embed)."
            self.status_label.configure(text=status_akhir, text_color="orange")
        
        self.batal_simpan_semua_wajah()

    def batal_simpan_semua_wajah(self):
        """Clear all face previews"""
        for widget in self.preview_frame_container.winfo_children():
            widget.destroy()
        self.preview_frame_container.pack_forget()
        self.detected_faces_data.clear()

    # --- ATTENDANCE FUNCTIONS ---
    def absensi_dari_gambar(self):
        """Absensi dari gambar yang diupload"""
        filepath = filedialog.askopenfilename(
            title="Pilih Gambar untuk Absensi Manual",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath: 
            return

        frame = cv2.imread(filepath)
        if frame is None:
            self.status_label.configure(text="Gagal membaca file gambar.", text_color="red")
            return

        self.status_label.configure(text="Memproses gambar unggahan...", text_color="cyan")
        self.update_idletasks()
        
        # Process image for attendance
        self._proses_gambar_untuk_review(frame)

    def _proses_gambar_untuk_review(self, frame):
        """Process image for attendance review"""
        # This will be handled by ReviewWindow class
        # For now, we'll keep the basic functionality
        faces = self.face_detector.detect_faces(frame, conf=0.4)
        
        if len(faces) == 0:
            self.status_label.configure(text="Tidak ada wajah terdeteksi pada gambar ini.", text_color="orange")
            return
            
        # Create review window
        ReviewWindow(self, frame, self.face_detector, self.face_recognizer, 
                    self.db_manager, self.attendance_logger)

    def mulai_absensi_terjadwal_kamera(self):
        """Start scheduled attendance with camera"""
        if self.camera_index == -1:
            self.status_label.configure(text="Pilih kamera yang valid terlebih dahulu.", text_color="orange")
            return
            
        if not self.db_manager.embedding_db:
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

        self.timer_interval_ms = interval_menit * 60 * 1000  # Convert to milliseconds
        self.current_repetition = 0

        # Disable buttons during timer
        self.start_timer_button.configure(state="disabled", text="Timer Kamera Berjalan...")
        self.upload_absen_button.configure(state="disabled")
        self.tambah_data_button.configure(state="disabled")
        self.use_camera_checkbox.configure(state="disabled")

        self.status_label.configure(
            text=f"Absensi kamera dimulai. Menunggu {interval_menit} menit...", 
            text_color="cyan"
        )
        self._jalankan_timer_kamera()

    def _jalankan_timer_kamera(self):
        """Run camera timer"""
        if self.current_repetition < self.total_repetitions:
            self.current_repetition += 1
            status_msg = f"Menunggu interval {self.current_repetition}/{self.total_repetitions} ({int(self.timer_interval_ms/60000)} menit)..."
            self.status_label.configure(text=status_msg, text_color="cyan")
            self.timer_job_id = self.after(self.timer_interval_ms, self._capture_and_process_kamera)
        else:
            self._selesai_timer_kamera()

    def _capture_and_process_kamera(self):
        """Capture and process image from camera"""
        self.status_label.configure(
            text=f"Mengambil gambar ({self.current_repetition}/{self.total_repetitions}) dari Kamera {self.camera_index}...", 
            text_color="yellow"
        )
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

        # Schedule next timer
        if self.current_repetition < self.total_repetitions:
            self._jalankan_timer_kamera()
        else:
            self._selesai_timer_kamera()

    def _selesai_timer_kamera(self):
        """Finish camera timer"""
        if self.timer_job_id:
            self.after_cancel(self.timer_job_id)
            self.timer_job_id = None
        
        # Re-enable buttons
        self.start_timer_button.configure(state="normal", text="Mulai Absen Terjadwal Kamera")
        self.upload_absen_button.configure(state="normal")
        self.tambah_data_button.configure(state="normal")
        self.use_camera_checkbox.configure(state="normal")
        
        if self.current_repetition >= self.total_repetitions:
            self.status_label.configure(text="Absensi terjadwal selesai.", text_color="lightgreen")

    # --- DATABASE MANAGEMENT ---
    def tampilkan_database(self):
        """Show database management window"""
        if self.db_window is not None and self.db_window.winfo_exists():
            self.db_window.focus()
            return

        self.db_window = DatabaseWindow(self, self.db_manager)

    # --- UTILITY FUNCTIONS ---
    def mark_attendance(self, name, user_id, face_crop):
        """Mark attendance - wrapper for attendance logger"""
        success = self.attendance_logger.mark_attendance(name, user_id, face_crop)
        if success:
            self.after(0, lambda: self.status_label.configure(
                text=f"Absen Dicatat: {name} | {datetime.now().strftime('%H:%M:%S')}", 
                text_color="lightgreen"
            ))
        return success