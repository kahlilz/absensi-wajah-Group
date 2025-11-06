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
                face_image, 
                model_name='SFace', 
                enforce_detection=False, 
                detector_backend='skip'
            )
            return embedding_obj[0]['embedding']
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def compare_embeddings(self, embedding1, embedding2):
        """Membandingkan dua embedding"""
        try:
            distance = cosine(embedding1, embedding2)
            return distance < self.threshold, distance
        except Exception as e:
            print(f"Error comparing embeddings: {e}")
            return False, float('inf')
    
    def recognize_face(self, face_crop, embedding_db):
        """Mengenali wajah terhadap database"""
        # Implementation dari _recognize_one_face
        pass