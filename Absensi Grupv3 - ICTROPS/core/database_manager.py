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
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        return data
                    else:
                        print("File database tidak valid. Membuat baru.")
                        return {}
            except Exception as e:
                print(f"Error loading database: {e}")
                return {}
        print("File database tidak ditemukan. Membuat baru.")
        return {}
    
    def save_database(self):
        """Menyimpan database ke file"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.embedding_db, f)
            print("Database berhasil disimpan.")
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_user(self, user_id, name, embedding):
        """Menambah user baru ke database"""
        try:
            if user_id in self.embedding_db:
                # Update existing user
                if 'embeddings' not in self.embedding_db[user_id]:
                    # Convert from old format
                    old_embedding = self.embedding_db[user_id].get('embedding')
                    self.embedding_db[user_id]['embeddings'] = [old_embedding] if old_embedding else []
                    if 'embedding' in self.embedding_db[user_id]:
                        del self.embedding_db[user_id]['embedding']
                
                self.embedding_db[user_id]['embeddings'].append(embedding)
                self.embedding_db[user_id]['name'] = name
            else:
                # Create new user
                self.embedding_db[user_id] = {
                    'name': name,
                    'embeddings': [embedding]
                }
            return True
        except Exception as e:
            print(f"Error adding user: {e}")
            return False
    
    def delete_user(self, user_id):
        """Menghapus user dari database"""
        if user_id in self.embedding_db:
            del self.embedding_db[user_id]
            return True
        return False
    
    def get_all_users(self):
        """Mendapatkan semua data user"""
        return list(self.embedding_db.items())