# Fruit Detection (YOLOv8) - Wrapstation Technical Test

Repositori ini berisi solusi untuk **Soal No. 1 (AI Training & Object Detection)** dari uji teknis Wrapstation Fullstack Developer. Model dilatih menggunakan arsitektur YOLOv8 (dari library `ultralytics`) untuk mendeteksi objek buah-buahan.

## Prasyarat (Requirements)
Pastikan sistem Anda sudah terinstal:
- **Python** versi 3.8 atau lebih baru.
- Koneksi internet untuk mengunduh pretrained weights YOLOv8.

## Instalasi Dependensi
Untuk menginstal semua pustaka yang dibutuhkan, jalankan perintah berikut di terminal:
```bash
pip install ultralytics opencv-python
```
*(Catatan: Jika di Windows Anda menggunakan alias `py`, silakan gunakan perintah `py -m pip install ultralytics opencv-python`).*

## Struktur Direktori
- `train.py`: Script untuk melakukan inisialisasi dan melatih model cerdas (YOLOv8).
- `inference.py`: Script untuk menjalankan *inference* (deteksi objek) secara *real-time* atau pada *test images* dan menampilkannya di layar (menggunakan `cv2.imshow`).
- `dataset.yaml`: Konfigurasi file dataset YOLO yang mendefinisikan lokasi file gambar dan jumlah *class*.

## Cara Menjalankan Script
### 1. Training Model
**PENTING**: Sebelum menjalankan proses training, pastikan dataset yang digunakan adalah dataset berformat **YOLO Object Detection** (memiliki file gambar `.jpg` beserta file pasangan anotasi `.txt` untuk *bounding box* di dalam folder `train/labels` dan `valid/labels`). 

Dataset yang tercantum pada soal (di Kaggle) berformat *Multi-Class Classification* sehingga tidak dapat digunakan untuk melatih sistem *Object Detection* tanpa dilakukan konversi anotasi terlebih dahulu.

Untuk memulai proses training, jalankan:
```bash
python train.py
```
Model terbaik hasil training (`best.pt`) akan otomatis tersimpan di dalam folder `runs/detect/fruit_detection_model/weights/`.

### 2. Menjalankan Inference (Deteksi Objek)
Setelah file `best.pt` berhasil dibuat dari proses training, Anda bisa menjalankan script inference:
```bash
python inference.py
```
Script ini akan membuka sebuah *pop-up window* (berbasis OpenCV) yang menampilkan gambar hasil deteksi (kotak *bounding box*, nama objek, dan akurasi/probabilitas).
- Tekan tombol sembarang untuk beralih ke gambar berikutnya.
- Tekan tombol `q` untuk keluar dari program.
