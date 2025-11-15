from sklearn.datasets import fetch_lfw_people
import numpy as np

# Ambil dataset
data = fetch_lfw_people(min_faces_per_person=0, resize=1)

# Ambil label orangnya
labels = data.target
names = data.target_names

# Hitung jumlah foto per identitas
unique, counts = np.unique(labels, return_counts=True)

print("Total orang:", len(unique))
print("Total gambar:", len(labels))
print("\nJumlah foto per orang:\n")

for name, count in zip(names, counts):
    print(f"{name}: {count}")
