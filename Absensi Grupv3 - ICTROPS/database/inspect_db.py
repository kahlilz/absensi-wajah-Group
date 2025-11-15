import pickle
import numpy as np

# Gunakan 'r' untuk raw string, ini memperbaiki error path Windows Anda
NAMA_FILE_PKL = r"D:\kahlil\Kuliah\Skripsi\Project\Windows\1. absensi-wajah-Group\Absensi Grupv3\database\embeddings.pkl" 

print(f"Membuka file: {NAMA_FILE_PKL}...\n")

try:
    with open(NAMA_FILE_PKL, 'rb') as f:
        data = pickle.load(f)

    print("--- Berhasil Memuat Data ---")
    
    # Asumsi data adalah dictionary
    if not isinstance(data, dict) or len(data) == 0:
        print("Error: Data bukan dictionary atau dictionary kosong.")
        exit()

    # Ambil kunci pertama untuk dijadikan sampel
    kunci_pertama = list(data.keys())[0]
    # 'data_user' adalah KUMPULAN (list) dari 18 embeddings
    data_user = data[kunci_pertama] 

    print(f"Menganalisis data untuk Kunci/ID: '{kunci_pertama}'")
    
    # Cek apakah 'data_user' adalah list (atau numpy array)
    if isinstance(data_user, (list, np.ndarray)) and len(data_user) > 0:
        
        print(f"Tipe data yang tersimpan untuk user ini: {type(data_user)}")
        print(f"Jumlah embedding tersimpan: {len(data_user)}") # Ini akan mencetak 18

        # --- INI BAGIAN PENTINGNYA ---
        # Ambil embedding PERTAMA dari 18 embedding itu
        vektor_tunggal = data_user[0]
        
        print("\n--- Menganalisis 1 Vektor Embedding Saja ---")
        print(f"Tipe data 1 vektor: {type(vektor_tunggal)}")
        
        # Ini akan menghitung jumlah angka di dalam vektor tunggal itu
        if hasattr(vektor_tunggal, '__len__'):
            print(f"\n===> DIMENSI VEKTOR TUNGGAL: {len(vektor_tunggal)}") # Ini seharusnya 128
        
        print("\nPratinjau 1 Vektor (hanya 10 angka pertama):")
        if isinstance(vektor_tunggal, np.ndarray):
            print(f"{vektor_tunggal.tolist()[:10]}...")
        else:
            print(f"{vektor_tunggal[:10]}...")

    else:
        print(f"Struktur data tidak terduga. Data: {data_user}")

except FileNotFoundError:
    print(f"ERROR: File tidak ditemukan di '{NAMA_FILE_PKL}'")
except Exception as e:
    print(f"Terjadi error saat membaca file: {e}")