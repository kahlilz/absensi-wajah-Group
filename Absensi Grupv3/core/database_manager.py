import pickle
import os
from core.utils import get_app_path

class DatabaseManager:
    def __init__(self):
        self.app_path = get_app_path()
        self.db_path = os.path.join(self.app_path, "database", "embeddings.pkl")
        self.embedding_db = self.load_database()
    
    def load_database(self):
        """Memuat database dari file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
        return {}
    
    def save_database(self):
        """Menyimpan database ke file"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.embedding_db, f)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_user(self, user_id, name, embedding):
        """Menambah user baru ke database"""
        # Implementation dari simpan_dan_buat_embedding
        pass
    
    def delete_user(self, user_id):
        """Menghapus user dari database"""
        # Implementation dari hapus_user_dari_db
        pass
    
    def get_all_users(self):
        """Mendapatkan semua data user"""
        return self.embedding_db.items()