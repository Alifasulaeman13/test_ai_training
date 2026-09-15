# Soal 1: AI Training (Fruit Detection)

Repositori ini berisi pengerjaan Soal 1 untuk posisi Full Stack Developer di Wrapstation.

Terdapat perbedaan antara instruksi tes (diminta *Object Detection* dengan *bounding box*) dan dataset Kaggle yang diberikan (dataset *Image Classification* tanpa *bounding box*). Oleh karena itu, saya menyediakan dua opsi script di repo ini agar keduanya tetap bisa diuji menggunakan YOLOv8.

## System Specs
- OS: Windows
- CPU: Intel Core i7 Gen 14
- RAM: 16 GB
- Storage: SSD 512GB Gen4

## Persiapan
Install library yang dibutuhkan:
```bash
pip install ultralytics opencv-python pandas
```

---

## Opsi 1: Object Detection (Sesuai Syarat PDF)
Script ini dibuat murni untuk memenuhi instruksi soal yang meminta sistem *Object Detection* yang menampilkan *bounding box*.

**Cara test:**
```bash
python train.py
python inference.py
```
*Note: Karena dataset Kaggle yang dilampirkan berformat klasifikasi (tidak ada anotasi koordinat `.txt`), proses training di script ini akan memunculkan error `No labels found`. Secara code, logic deteksi dan rendering-nya sudah jalan dan tinggal dipakaikan dataset YOLO detection yang valid.*

---

## Opsi 2: Image Classification (Menyesuaikan Dataset Kaggle)
Script alternatif ini saya buat menyesuaikan format asli dari dataset Kaggle yang dilampirkan di PDF (klasifikasi gambar tanpa *bounding box*).

Model hasil training klasifikasinya sudah saya sertakan di repo ini (`best.pt`), jadi **bisa langsung di-test tanpa harus training dari awal.**

**Cara melihat hasil klasifikasi (Bisa langsung di-run):**
```bash
python inference_cls.py
```
- Tekan sembarang tombol untuk melihat gambar selanjutnya.
- Tekan `q` untuk keluar dari window OpenCV.

*(Opsional) Jika ingin mencoba proses training ulang dari awal:*
1. Run `python prepare_classification_dataset.py` (untuk merapikan flat folder gambar ke dalam masing-masing class folder).
2. Run `python train_cls.py` (untuk memulai proses training klasifikasi YOLO).
