import customtkinter as ctk
from PIL import Image
import cv2
import numpy as np

class ReviewWindow:
    def __init__(self, parent, frame, face_detector, face_recognizer, db_manager, attendance_logger):
        self.parent = parent
        self.frame = frame
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.db_manager = db_manager
        self.attendance_logger = attendance_logger
        
        self.setup_window()
        self.process_frame()

    def setup_window(self):
        """Setup window review absensi"""
        self.window = ctk.CTkToplevel(self.parent)
        self.window.title("Menu Review Absensi")
        self.window.geometry("600x700")
        self.window.transient(self.parent)
        self.window.attributes('-topmost', True)

        self.scroll_frame = ctk.CTkScrollableFrame(
            self.window, 
            label_text="Hasil Deteksi - Konfirmasi Absensi di Bawah"
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        close_button = ctk.CTkButton(
            self.window, 
            text="Tutup Review", 
            command=self.window.destroy
        )
        close_button.pack(pady=10)

    def process_frame(self):
        """Process frame and display results"""
        THRESHOLD_YAKIN = 0.2
        absen_otomatis_count = 0

        # Detect faces
        faces = self.face_detector.detect_faces(self.frame, conf=0.4)

        if len(faces) == 0:
            self.parent.status_label.configure(
                text="Tidak ada wajah terdeteksi pada gambar ini.", 
                text_color="orange"
            )
            return

        # Prepare name options
        pilihan_nama = {"Tidak Dikenal": "Tidak Dikenal"}
        for user_id, data in self.db_manager.get_all_users():
            nama = data.get('name')
            if isinstance(nama, str):
                pilihan_nama[nama] = user_id

        # Process each face
        for i, box in enumerate(faces):
            x1, y1, x2, y2 = box
            # Ensure valid coordinates
            y1, y2 = max(0, y1), min(self.frame.shape[0], y2)
            x1, x2 = max(0, x1), min(self.frame.shape[1], x2)
            face_crop = self.frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # Recognize face
            recognized_id, min_distance = self.recognize_face(face_crop)
            
            # Create card for each face
            self.create_face_card(
                i, face_crop, recognized_id, min_distance, 
                pilihan_nama, THRESHOLD_YAKIN
            )

            # Auto-attendance for confident matches
            if (recognized_id not in ["Tidak Dikenal", "Crop Gagal", "Error Proses", "DB Kosong"] and 
                min_distance < THRESHOLD_YAKIN):
                nama_dikenali = self.db_manager.embedding_db.get(recognized_id, {}).get('name', 'Error ID')
                self.attendance_logger.mark_attendance(nama_dikenali, recognized_id, face_crop)
                absen_otomatis_count += 1

        status_text = f"Review Selesai! {absen_otomatis_count} absen otomatis tercatat."
        if hasattr(self.parent, 'timer_job_id') and self.parent.timer_job_id is None:
            self.parent.status_label.configure(text=status_text, text_color="white")

    def recognize_face(self, face_crop):
        """Recognize a single face"""
        if face_crop is None or face_crop.size == 0:
            return "Crop Gagal", float('inf')
        
        try:
            embedding = self.face_recognizer.extract_embedding(face_crop)
            if embedding is None:
                return "Error Proses", float('inf')
            
            if not self.db_manager.embedding_db:
                return "DB Kosong", float('inf')

            min_distance = float("inf")
            recognized_id = "Tidak Dikenal"

            for user_id, data in self.db_manager.get_all_users():
                embeddings_to_check = []
                if 'embeddings' in data and isinstance(data['embeddings'], list):
                    embeddings_to_check = data['embeddings']
                elif 'embedding' in data:
                    embeddings_to_check = [data['embedding']]
                
                if not embeddings_to_check:
                    continue

                for db_embedding in embeddings_to_check:
                    if db_embedding is not None and len(db_embedding) > 0:
                        try:
                            is_match, distance = self.face_recognizer.compare_embeddings(
                                embedding, db_embedding
                            )
                            if distance < min_distance:
                                min_distance = distance
                                recognized_id = user_id
                        except ValueError as ve:
                            print(f"Error cosine distance untuk ID {user_id}: {ve}")
                            continue

            # Validate recognized_id exists in database
            if recognized_id != "Tidak Dikenal" and recognized_id not in self.db_manager.embedding_db:
                recognized_id = "Tidak Dikenal"
                min_distance = float('inf')

            return recognized_id, min_distance

        except Exception as e:
            print(f"Error recognizing face: {e}")
            return "Error Proses", float('inf')

    def create_face_card(self, index, face_crop, recognized_id, min_distance, pilihan_nama, threshold):
        """Create UI card for each detected face"""
        card_frame = ctk.CTkFrame(self.scroll_frame, border_width=1)
        card_frame.pack(fill="x", padx=10, pady=10)

        # Face preview
        try:
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            ctk_image = ctk.CTkImage(light_image=face_pil, dark_image=face_pil, size=(100, 100))
            preview_label = ctk.CTkLabel(card_frame, image=ctk_image, text="")
            preview_label.pack(side="left", padx=10, pady=10)
        except Exception as e:
            print(f"Error creating preview: {e}")
            preview_label = ctk.CTkLabel(card_frame, text="Gagal Load Preview", width=100, height=100)
            preview_label.pack(side="left", padx=10, pady=10)

        info_frame = ctk.CTkFrame(card_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=10)

        is_confident = False
        nama_dikenali = "N/A"
        
        if recognized_id not in ["Tidak Dikenal", "Crop Gagal", "Error Proses", "DB Kosong"]:
            if recognized_id in self.db_manager.embedding_db:
                if min_distance < threshold:
                    is_confident = True
                nama_dikenali = self.db_manager.embedding_db.get(recognized_id, {}).get('name', 'Error ID')
            else:
                recognized_id = "Tidak Dikenal"
                min_distance = float('inf')

        if is_confident:
            # Auto-attendance case (already handled in process_frame)
            keyakinan = (1 - min_distance) * 100
            info_text = f"Nama: {nama_dikenali}\nID: {recognized_id}\nKeyakinan: {keyakinan:.1f}%"
            info_label = ctk.CTkLabel(info_frame, text=info_text, justify="left")
            info_label.pack(anchor="w", pady=5)

            status_label = ctk.CTkLabel(
                info_frame, 
                text="✅ Absen Otomatis Tercatat", 
                font=ctk.CTkFont(weight="bold"), 
                text_color="lightgreen"
            )
            status_label.pack(anchor="w", pady=10)
        else:
            # Manual confirmation case
            saran_sistem = "Saran: Tidak Dikenal"
            nama_saran = "Tidak Dikenal"
            keyakinan = 0.0
            
            if recognized_id not in ["Tidak Dikenal", "Crop Gagal", "Error Proses", "DB Kosong"]:
                nama_saran = nama_dikenali
                keyakinan = (1 - min_distance) * 100
                saran_sistem = f"Saran: {nama_saran} (Yakin: {keyakinan:.0f}%)"

            saran_label = ctk.CTkLabel(info_frame, text=saran_sistem, font=ctk.CTkFont(size=12))
            saran_label.pack(anchor="w", pady=(5,0))

            # Name selection combobox
            nama_combobox = ctk.CTkComboBox(
                info_frame, 
                values=list(pilihan_nama.keys()), 
                width=200
            )
            nama_combobox.set(nama_saran if nama_saran in pilihan_nama else "Tidak Dikenal")
            nama_combobox.pack(anchor="w", pady=5)

            # Confirm button
            konfirmasi_button = ctk.CTkButton(
                info_frame, 
                text="Konfirmasi & Catat Absen",
                command=lambda: self.handle_konfirmasi_absen(
                    pilihan_nama.get(nama_combobox.get()), 
                    nama_combobox.get(), 
                    konfirmasi_button, 
                    face_crop.copy()
                )
            )
            konfirmasi_button.pack(anchor="w", pady=10)

    def handle_konfirmasi_absen(self, user_id_terpilih, nama_terpilih, button, face_crop):
        """Handle manual attendance confirmation"""
        if not user_id_terpilih or user_id_terpilih == "Tidak Dikenal":
            self.parent.status_label.configure(
                text="Pilihan tidak valid untuk absen.", 
                text_color="orange"
            )
            return

        # Convert to int if needed
        if not isinstance(user_id_terpilih, int):
            try:
                user_id_terpilih = int(user_id_terpilih)
            except ValueError:
                self.parent.status_label.configure(
                    text="ID User terpilih tidak valid.", 
                    text_color="red"
                )
                return

        # Mark attendance
        self.attendance_logger.mark_attendance(nama_terpilih, user_id_terpilih, face_crop)
        
        # Active learning - update database
        try:
            if face_crop is None or face_crop.size == 0:
                raise ValueError("Data wajah tidak valid untuk embedding.")

            embedding = self.face_recognizer.extract_embedding(face_crop)
            if embedding is None:
                raise ValueError("Gagal mengekstrak embedding.")

            # Update database with new embedding
            if user_id_terpilih in self.db_manager.embedding_db:
                if 'embeddings' not in self.db_manager.embedding_db[user_id_terpilih]:
                    self.db_manager.embedding_db[user_id_terpilih]['embeddings'] = []
                
                self.db_manager.embedding_db[user_id_terpilih]['embeddings'].append(embedding)
                
                # Update name if different
                if self.db_manager.embedding_db[user_id_terpilih]['name'] != nama_terpilih:
                    self.db_manager.embedding_db[user_id_terpilih]['name'] = nama_terpilih
            else:
                # Create new entry
                self.db_manager.embedding_db[user_id_terpilih] = {
                    'name': nama_terpilih, 
                    'embeddings': [embedding]
                }
            
            self.db_manager.save_database()
            print(f"Active Learning Sukses! Database untuk '{nama_terpilih}' diperbarui.")

        except Exception as e:
            print(f"Error saat Active Learning: {e}")
            self.parent.status_label.configure(
                text=f"Absen tercatat, tapi gagal update DB: {e}", 
                text_color="orange"
            )

        button.configure(text="Tercatat & Terupdate", state="disabled", fg_color="grey")