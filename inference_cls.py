from ultralytics import YOLO
import cv2
import os
import glob

# Path ke direktori gambar untuk testing (bebas ambil dari mana saja yang ada .jpg-nya)
TEST_IMAGES_DIR = "dataset/test"

# Path ke model hasil training (Weights) klasifikasi
MODEL_PATH = r"runs\classify\fruit_classification_model\weights\best.pt"

def main():
    # Pastikan model (best.pt) sudah ada hasil dari script training klasifikasi
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model belum ditemukan di {MODEL_PATH}.")
        print("Harap jalankan train_cls.py terlebih dahulu.")
        return

    # Load trained model klasifikasi
    model = YOLO(MODEL_PATH)

    # Ambil beberapa gambar dari folder test untuk dilakukan inferensi
    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
    
    if not image_paths:
        print(f"Error: Tidak ada gambar .jpg ditemukan di direktori {TEST_IMAGES_DIR}")
        return

    print("Memulai preview klasifikasi gambar (Tekan sembarang tombol pada window untuk lanjut, atau 'q' untuk keluar)...")

    for img_path in image_paths:
        # Menjalankan inferensi klasifikasi
        results = model(img_path)

        for result in results:
            # Mengambil gambar yang sudah dilengkapi dengan teks prediksi kelas (tanpa kotak)
            annotated_frame = result.plot()

            # (Syarat: Script harus menampilkan pop-up window saat dijalankan)
            cv2.imshow("YOLO Fruit Classification Preview", annotated_frame)
            
            # Tunggu input keyboard
            key = cv2.waitKey(0) & 0xFF
            
            # Jika user menekan tombol 'q', keluar dari perulangan
            if key == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
