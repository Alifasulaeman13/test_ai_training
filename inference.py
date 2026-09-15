from ultralytics import YOLO
import cv2
import os
import glob

# Path ke direktori gambar untuk testing
# (Sesuai syarat: path direktori gambar ditentukan di dalam script)
TEST_IMAGES_DIR = r"d:\project\test-ai-training\Fruits by YOLO\Fruits by YOLO\test"

# Path ke model hasil training (Weights)
MODEL_PATH = r"runs\detect\fruit_detection_model\weights\best.pt"

def main():
    # Pastikan model (best.pt) sudah ada hasil dari script training
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model belum ditemukan di {MODEL_PATH}.")
        print("Harap jalankan train.py terlebih dahulu, atau pastikan dataset YOLO (dengan file .txt koordinat) sudah benar.")
        return

    # Load trained model
    model = YOLO(MODEL_PATH)

    # Ambil beberapa gambar dari folder test untuk dilakukan inferensi
    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    
    if not image_paths:
        print(f"Error: Tidak ada gambar .jpg ditemukan di direktori {TEST_IMAGES_DIR}")
        return

    print("Memulai preview deteksi gambar (Tekan sembarang tombol pada window untuk lanjut, atau 'q' untuk keluar)...")

    for img_path in image_paths:
        # Menjalankan inferensi
        results = model(img_path)

        for result in results:
            # Mengambil gambar yang sudah dilengkapi dengan anotasi deteksi dan label kelas
            annotated_frame = result.plot()

            # (Syarat: Script harus menampilkan pop-up window saat dijalankan)
            cv2.imshow("YOLO Fruit Detection Preview", annotated_frame)
            
            # Tunggu input keyboard
            key = cv2.waitKey(0) & 0xFF
            
            # Jika user menekan tombol 'q', keluar dari perulangan
            if key == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
