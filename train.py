from ultralytics import YOLO
import os

# Tentukan path direktori gambar/dataset di dalam script
# (Syarat: file path direktori gambar ditentukan dalam script, tidak perlu input terminal)
# Gunakan absolute path ke file dataset.yaml
DATASET_YAML_PATH = r"d:\project\test-ai-training\task1_ai_training\dataset.yaml"

def main():
    print("Memulai proses training YOLO...")
    
    # 1. Inisialisasi model YOLO dasar (pre-trained model)
    model = YOLO("yolov8n.pt") 

    # 2. Melakukan training model menggunakan dataset
    # epochs diset kecil (misal 10) untuk keperluan test, bisa ditingkatkan untuk akurasi lebih baik
    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=10,
        imgsz=640,
        name="fruit_detection_model",
        device="cpu" # Ubah ke 0 jika menggunakan GPU
    )

    print("\nTraining selesai!")
    print("Model/Weights tersimpan di direktori: runs/detect/fruit_detection_model/weights/best.pt")

if __name__ == '__main__':
    main()
