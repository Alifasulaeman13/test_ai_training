from ultralytics import YOLO
import os

# Path ke direktori dataset klasifikasi (sudah dipisah per folder)
# Sesuai syarat: path direktori ditentukan dalam script
DATASET_DIR = r"d:\project\test-ai-training\task1_ai_training\dataset_classification"

def main():
    print("Memulai proses training YOLOv8 Classification...")
    
    # Pastikan dataset klasifikasi sudah di-generate
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset {DATASET_DIR} tidak ditemukan!")
        print("Harap jalankan prepare_classification_dataset.py terlebih dahulu.")
        return

    # 1. Inisialisasi model YOLO Klasifikasi (pre-trained model)
    model = YOLO("yolov8n-cls.pt") 

    # 2. Melakukan training model klasifikasi
    results = model.train(
        data=DATASET_DIR,
        epochs=10,
        imgsz=224, # Ukuran gambar standar untuk klasifikasi
        name="fruit_classification_model",
        device="cpu" # Ubah ke 0 jika menggunakan GPU
    )

    print("\nTraining Klasifikasi Selesai!")
    print("Model/Weights tersimpan di direktori: runs/classify/fruit_classification_model/weights/best.pt")

if __name__ == '__main__':
    main()
