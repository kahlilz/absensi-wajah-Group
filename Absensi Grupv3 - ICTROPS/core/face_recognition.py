import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

class FaceRecognizer:
    def __init__(self, threshold=0.45):
        self.threshold = threshold
    
    def extract_embedding(self, face_image):
        """Mengekstrak embedding dari crop wajah"""
        try:
            embedding_obj = DeepFace.represent(
                img_path=face_image,
                model_name='SFace', 
                enforce_detection=False, 
                detector_backend='skip'
            )
            if embedding_obj and len(embedding_obj) > 0:
                return embedding_obj[0]['embedding']
            return None
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def compare_embeddings(self, embedding1, embedding2):
        """Membandingkan dua embedding"""
        try:
            if embedding1 is None or embedding2 is None:
                return False, float('inf')
                
            distance = cosine(embedding1, embedding2)
            return distance < self.threshold, distance
        except Exception as e:
            print(f"Error comparing embeddings: {e}")
            return False, float('inf')