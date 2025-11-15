import cv2
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ================== KONFIGURASI ==================
# GANTI PATH INI DENGAN LOKASI FILE ANDA
YOLOv8_MODEL = r"D:\kahlil\Kuliah\Skripsi\Project\Windows\1. absensi-wajah-Group\Absensi Grupv3 - ICTROPS\yolov8n-face-lindevs.pt"  # GANTI!
INPUT_FOLDER = r"D:\kahlil\Kuliah\Skripsi\Project\Windows\1. absensi-wajah-Group\Absensi Grupv3 - ICTROPS\Gambar Test\Foto"  # GANTI!
OUTPUT_FOLDER = "yolov8_grouped_results"
CONFIDENCE_THRESHOLD = 0.5  # Ubah ini untuk lebih ketat/longgar
# =================================================

def main():
    print("🚀 Starting YOLOv8 Photo Grouper...")
    
    # Validasi file model
    if not os.path.exists(YOLOv8_MODEL):
        print(f"❌ Error: Model file tidak ditemukan: {YOLOv8_MODEL}")
        return
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Folder input tidak ditemukan: {INPUT_FOLDER}")
        return
    
    # Load YOLOv8 model
    print("🔄 Loading YOLOv8 model...")
    try:
        model = YOLO(YOLOv8_MODEL)
        print(f"✅ Model loaded: {os.path.basename(YOLOv8_MODEL)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create output directory
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Statistics
    total_photos = 0
    total_faces = 0
    
    print(f"\n📁 Processing folder: {INPUT_FOLDER}")
    
    # Process images
    for filename in os.listdir(INPUT_FOLDER):
        filepath = os.path.join(INPUT_FOLDER, filename)
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in image_extensions:
            continue
            
        total_photos += 1
        print(f"\n🔍 Processing: {filename}")
        
        try:
            # Detect faces with YOLOv8
            results = model(filepath, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            face_count = 0
            for result in results:
                face_count = len(result.boxes)
            
            print(f"   ✅ Detected: {face_count} face(s)")
            total_faces += face_count
            
            # Group by face count
            group_folder = os.path.join(OUTPUT_FOLDER, f"{face_count}_faces")
            Path(group_folder).mkdir(exist_ok=True)
            
            # Copy file to appropriate group
            shutil.copy2(filepath, os.path.join(group_folder, filename))
            
            # Save annotated version
            if face_count > 0:
                annotated_img = results[0].plot()
                annotated_filename = f"annotated_{filename}"
                cv2.imwrite(os.path.join(group_folder, annotated_filename), annotated_img)
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Put error files in separate folder
            error_folder = os.path.join(OUTPUT_FOLDER, "errors")
            Path(error_folder).mkdir(exist_ok=True)
            shutil.copy2(filepath, os.path.join(error_folder, filename))
    
    # Print summary
    print("\n" + "="*50)
    print("📊 FINAL SUMMARY")
    print("="*50)
    print(f"Total photos processed: {total_photos}")
    print(f"Total faces detected: {total_faces}")
    if total_photos > 0:
        print(f"Average faces per photo: {total_faces/total_photos:.1f}")
    print(f"Results saved in: {OUTPUT_FOLDER}")
    print("="*50)

if __name__ == "__main__":
    main()