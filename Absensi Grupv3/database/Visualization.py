import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_embeddings():
    """Load embeddings from pickle file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'embeddings.pkl')
    
    if not os.path.exists(file_path):
        print(f"File tidak ditemukan: {file_path}")
        return None
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data loaded: {len(data)} users")
    return data

def visualize_embeddings_tsne(data):
    """Visualize embeddings using TSNE"""
    
    # Ekstrak semua embeddings
    all_embeddings = []
    user_ids = []
    user_names = []
    
    for user_id, user_data in data.items():
        name = user_data.get('name', f'User_{user_id}')
        
        if 'embeddings' in user_data:
            for emb in user_data['embeddings']:
                all_embeddings.append(emb)
                user_ids.append(user_id)
                user_names.append(name)
        elif 'embedding' in user_data:
            all_embeddings.append(user_data['embedding'])
            user_ids.append(user_id)
            user_names.append(name)
    
    if not all_embeddings:
        print("Tidak ada embeddings ditemukan!")
        return
    
    all_embeddings = np.array(all_embeddings)
    print(f"Total embeddings: {len(all_embeddings)}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    
    # TSNE
    n_samples = len(all_embeddings)
    perplexity = min(30, n_samples - 1)
    
    print(f"Running TSNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Buat color map untuk setiap user
    unique_users = list(set(user_ids))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_users)))
    user_to_color = {uid: colors[i] for i, uid in enumerate(unique_users)}
    
    for i, (x, y) in enumerate(reduced):
        color = user_to_color[user_ids[i]]
        plt.scatter(x, y, c=[color], s=50, alpha=0.7)
        plt.annotate(user_names[i], (x, y), fontsize=8, alpha=0.7)
    
    plt.title(f'TSNE Visualization of Face Embeddings ({n_samples} samples)')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = load_embeddings()
    if data:
        visualize_embeddings_tsne(data)