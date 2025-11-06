import os
import cv2
from datetime import datetime
from core.utils import get_app_path

class AttendanceLogger:
    def __init__(self):
        self.app_path = get_app_path()
        self.log_dir = os.path.join(self.app_path, "log_absensi")
        self.photo_dir = os.path.join(self.log_dir, "foto_log")
        self.csv_path = os.path.join(self.log_dir, "Attendance.csv")
        
        # Create directories if they don't exist
        os.makedirs(self.photo_dir, exist_ok=True)
    
    def mark_attendance(self, name, user_id, face_crop):
        """Mencatat presensi"""
        try:
            # Save photo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            photo_filename = f"{user_id}_{timestamp}.jpg"
            photo_path = os.path.join(self.photo_dir, photo_filename)
            
            if face_crop is not None and face_crop.size > 0:
                save_success = cv2.imwrite(photo_path, face_crop)
                if not save_success:
                    photo_path = "N/A (Gagal simpan foto)"
            else:
                photo_path = "N/A (Data wajah tidak valid)"

            # Log to CSV
            date_str = datetime.now().strftime('%Y-%m-%d')
            time_str = datetime.now().strftime('%H:%M:%S')
            
            file_exists = os.path.isfile(self.csv_path)
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                if not file_exists:
                    f.write("Nama;ID Siswa;Tanggal;Waktu;Path Foto\n")
                f.write(f"{name};{user_id};{date_str};{time_str};{photo_path}\n")
            
            print(f"Absen Dicatat: {name} | {time_str}")
            return True
            
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False