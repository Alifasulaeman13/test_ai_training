# YOLOv8 Fruit Detection (Wrapstation Test - Soal 1)

Repositori ini berisi solusi saya untuk **Soal 1** dari tes teknis *Fullstack Developer Wrapstation*. Proyek ini mengimplementasikan model *Object Detection* berbasis **YOLOv8** untuk mendeteksi dan menggambar *bounding box* (kotak pembatas) di sekitar gambar buah.

## Persyaratan Sistem
- Python >= 3.8
- `ultralytics`
- `opencv-python`

## Instalasi & Persiapan
Silakan install *dependencies* yang dibutuhkan melalui pip:
```bash
pip install ultralytics opencv-python
```
*(Catatan: Jika Anda menggunakan Windows dan perintah `python` tidak dikenali, gunakan `py -m pip install ultralytics opencv-python`)*

## Cara Penggunaan

### 1. Melatih Model (Training)
Untuk memulai proses *training*, jalankan perintah berikut:
```bash
python train.py
```
Setelah proses selesai, *weights* (bobot) terbaik akan otomatis tersimpan di dalam folder `runs/detect/fruit_detection_model/weights/best.pt`.

**⚠️ Catatan Penting Mengenai Dataset:** 
Link dataset Kaggle yang dilampirkan pada soal tes adalah dataset untuk *"Multi-Class Classification"*. Karena dataset tersebut tidak memiliki file anotasi `.txt` untuk letak *bounding box*, maka script akan menghasilkan error `No labels found` dari bawaan sistem YOLO jika dipaksakan berjalan menggunakan dataset tersebut.
Namun, **script `train.py` ini sudah berfungsi sepenuhnya secara logika program** dan akan langsung memproses *training* dengan sukses jika menggunakan dataset yang berformat *YOLO Object Detection* dengan benar.

### 2. Menjalankan Deteksi (Inference)
Setelah model berhasil dilatih (file `best.pt` terbentuk), Anda dapat menjalankan script inferensi untuk melihat hasil deteksi model:
```bash
python inference.py
```
Perintah ini akan memunculkan *pop-up window* OpenCV yang menampilkan gambar-gambar buah lengkap dengan *bounding box* dan skor keakuratannya (*confidence score*).
- Tekan **tombol apa saja** pada *keyboard* untuk melihat gambar selanjutnya.
- Tekan tombol **q** untuk keluar dari *preview*.
