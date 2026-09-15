# YOLOv8 Fruit Detection & Classification (Wrapstation Test - Soal 1)

Repositori ini berisi solusi saya untuk **Soal 1** dari tes teknis *Fullstack Developer Wrapstation*. Karena terdapat ambiguitas antara instruksi soal (meminta **Object Detection**) dan dataset Kaggle yang diberikan (dataset **Image Classification**), saya membuat **Dua Solusi (Opsi A & Opsi B)** untuk memastikan kedua skenario terjawab dengan sempurna menggunakan YOLOv8.

## Persyaratan Sistem
- Python >= 3.8
- `ultralytics`
- `opencv-python`
- `pandas` (hanya untuk Opsi B)

## 💻 Spesifikasi Sistem (Environment)
Sesuai dengan *Submission Guidelines*, berikut adalah spesifikasi sistem yang digunakan selama pengerjaan tugas ini:
- **OS:** Windows
- **Prosesor:** Intel Core i7 Gen 14
- **RAM:** 16 GB
- **Storage:** SSD 512GB Gen4

## Instalasi & Persiapan
Silakan install *dependencies* yang dibutuhkan melalui pip:
```bash
pip install ultralytics opencv-python pandas
```
*(Catatan: Jika Anda menggunakan Windows, gunakan `py -m pip install ...`)*

---

## 🛠️ OPSI A: Object Detection (Sesuai Syarat Teks PDF)
Opsi ini secara mutlak memenuhi instruksi PDF: *"Sistem Object Detection"* dan *"Gambar harus dilengkapi dengan anotasi deteksi (kotak/bounding box)"*.
- **Script Training:** `train.py`
- **Script Inference:** `inference.py`

**Cara Menjalankan:**
```bash
python train.py
python inference.py
```
*(Catatan: YOLO Object Detection wajib menggunakan dataset berformat kotak koordinat `.txt`. Karena dataset Kaggle yang dilampirkan tidak memiliki file koordinat, eksekusi menggunakan dataset tersebut akan mengeluarkan peringatan `No labels found`. Namun secara logika programming, script ini sudah 100% siap produksi jika diberikan dataset YOLO yang tepat).*

---

## 🛠️ OPSI B: Image Classification (Sesuai Dataset Kaggle)
Opsi ini dibuat sebagai alternatif jika pihak penilai ternyata memang bermaksud membuat AI *Image Classification* dengan mengikuti dataset Kaggle yang dilampirkan (tidak menggunakan kotak *bounding box*). YOLOv8 mendukung mode ini via model `yolov8n-cls.pt`.

**Langkah 1: Siapkan Struktur Data**
Dataset Kaggle berbentuk *flat* dengan file `_classes.csv`. Jalankan script ini agar dataset tersebut otomatis disusun ulang menjadi format folder kelas YOLO:
```bash
python prepare_classification_dataset.py
```

**Langkah 2: Melatih Model (Training)**
Latih model pengklasifikasi buah:
```bash
python train_cls.py
```
*Weights (bobot) terbaik akan tersimpan di folder `runs/classify/fruit_classification_model/weights/best.pt`.*

**Langkah 3: Menjalankan Deteksi (Inference)**
Lihat hasil AI menebak buah (teks label tanpa kotak):
```bash
python inference_cls.py
```
- Tekan **tombol apa saja** pada *keyboard* untuk melihat gambar selanjutnya.
- Tekan tombol **q** untuk keluar dari *preview window*.
