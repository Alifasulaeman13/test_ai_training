import os
import shutil
import pandas as pd

def prepare_split(split_name):
    # Path dataset asli (Kaggle)
    base_dir = "dataset"
    source_dir = os.path.join(base_dir, split_name)
    csv_file = os.path.join(source_dir, "_classes.csv")
    
    if not os.path.exists(csv_file):
        print(f"File {csv_file} tidak ditemukan, melewati {split_name}...")
        return

    # Path tujuan untuk dataset klasifikasi
    target_dir = os.path.join("dataset_classification", split_name)
    os.makedirs(target_dir, exist_ok=True)

    # Membaca CSV
    df = pd.read_csv(csv_file)
    # Bersihkan nama kolom (hilangkan spasi berlebih)
    df.columns = df.columns.str.strip()

    # Dapatkan daftar kelas (semua kolom kecuali 'filename')
    classes = df.columns[1:].tolist()

    # Buat folder untuk setiap kelas
    for cls in classes:
        os.makedirs(os.path.join(target_dir, cls), exist_ok=True)

    # Salin gambar ke folder masing-masing kelas
    for index, row in df.iterrows():
        filename = row['filename'].strip()
        source_path = os.path.join(source_dir, filename)
        
        if not os.path.exists(source_path):
            continue
            
        # Cari kelas yang bernilai 1
        for cls in classes:
            if row[cls] == 1:
                target_path = os.path.join(target_dir, cls, filename)
                shutil.copy2(source_path, target_path)
                break
                
    print(f"Selesai memproses dataset {split_name} -> {target_dir}")

def main():
    print("Menyiapkan dataset klasifikasi untuk YOLOv8...")
    prepare_split('train')
    prepare_split('valid')
    prepare_split('test')
    print("Persiapan dataset klasifikasi Selesai!")

if __name__ == '__main__':
    main()
