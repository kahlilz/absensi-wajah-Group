import cv2
from ultralytics import YOLO
from core.utils import resource_path

class FaceDetector:
    def __init__(self):
        model_path = resource_path('models/yolov8n-face-lindevs.pt')
        self.model = YOLO(model_path)
    
    def detect_faces(self, image, conf=0.4):
        """Mendeteksi wajah dalam gambar"""
        results = self.model(image, verbose=False, conf=conf)
        return results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    def detect_faces_visual(self, image, conf=0.5):
        """Deteksi untuk preview visual dengan bounding boxes"""
        results = self.model(image, stream=True, verbose=False, conf=conf)
        return results