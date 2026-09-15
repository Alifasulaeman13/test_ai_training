from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
from PIL import Image

# Path ke direktori gambar untuk testing (bebas ambil dari mana saja yang ada .jpg-nya)
TEST_IMAGES_DIR = "dataset/test"

# Path ke model hasil training (Weights) klasifikasi
MODEL_PATH = r"runs\classify\fruit_classification_model2\weights\best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r"runs\classify\fruit_classification_model\weights\best.pt"

def draw_custom_classification_card(img_bgr, result):
    """Menggambar kartu background gelap dengan teks kontras tinggi untuk prediksi klasifikasi."""
    if result.probs is None:
        return img_bgr

    top5_indices = result.probs.top5
    top5_confs = result.probs.top5conf.tolist()
    names = result.names

    lines = [f"{names[idx]}: {conf*100:.1f}%" for idx, conf in zip(top5_indices[:5], top5_confs[:5])]
    
    # Hitung lebar dan tinggi background box
    box_w = 230
    box_h = 28 * len(lines) + 15

    # Gambar overlay background hitam dengan border hijau
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 255, 128), 2)
    
    # Blending untuk efek background gelap semi-transparan
    cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0, img_bgr)

    y = 35
    for i, line in enumerate(lines):
        if i == 0:
            color = (0, 255, 128) # Hijau terang untuk Prediksi Utama (Top 1)
            thickness = 2
        else:
            color = (240, 240, 240) # Putih terang untuk alternatif
            thickness = 1
            
        cv2.putText(img_bgr, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness, cv2.LINE_AA)
        y += 26
        
    return img_bgr

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
        # Load menggunakan PIL Image untuk menghindari TypeError pada torchvision transforms (numpy ndarray)
        img_pil = Image.open(img_path)
        
        # Menjalankan inferensi klasifikasi
        results = model(img_pil)

        for result in results:
            # Konversi gambar ke OpenCV format BGR
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # Gambar teks prediksi dengan kartu latar belakang gelap kontras tinggi
            annotated_frame = draw_custom_classification_card(img_bgr, result)

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

