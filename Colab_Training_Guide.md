# 🚀 Panduan Training Sundanese Sequence OCR di Google Colab (GPU)

Panduan ini menjelaskan langkah demi langkah untuk melakukan training model CRNN+CTC Aksara Sunda secara instan di Google Colab memanfaatkan akselerasi GPU (gratis).

---

## 🛠️ Persiapan Awal
1. Buka [Google Colab](https://colab.research.google.com/).
2. Buat **Notebook Baru** (New Notebook).
3. Ubah runtime ke GPU:
   * Klik menu **Runtime** > **Change runtime type**.
   * Pilih **T4 GPU** pada kolom *Hardware accelerator*.
   * Klik **Save**.

---

## 📝 Langkah-Langkah (Salin-Tempel ke Sel Colab)

### Sel 1: Verifikasi Akses GPU
Jalankan perintah ini untuk memastikan GPU T4 aktif:
```bash
!nvidia-smi
```

### Sel 2: Clone Repositori Anda
Unggah repositori `OCR-CRNN-AksaraSunda` ke GitHub Anda, lalu jalankan perintah ini di Colab untuk mengunduh kodenya (ganti dengan URL repo GitHub Anda):
```bash
!git clone https://github.com/anoymdev/OCR-CRNN-AksaraSunda.git
%cd OCR-CRNN-AksaraSunda
```
*Alternatif tanpa GitHub*: Anda bisa mengompres folder `OCR-CRNN-AksaraSunda` menjadi `.zip`, mengunggahnya ke Google Drive Anda, lalu mount Drive di Colab dan unzip berkasnya.

### Sel 3: Install Dependensi
Kebutuhan pustaka python untuk training:
```bash
!pip install tensorflow pillow opencv-python-headless numpy scipy matplotlib scikit-learn
```

### Sel 4: Buat Dataset Sintetis Sundanese
Gunakan generator bawaan untuk membuat 15.000 sampel latih dan 2.000 sampel validasi (proses ini memakan waktu kurang dari 2 menit di server Google):
```bash
!python data/generate_sequence.py --script sunda --train-samples 15000 --val-samples 2000
```

### Sel 5: Mulai Proses Training dengan GPU
Jalankan skrip training. Karena menggunakan GPU T4, setiap epoch akan selesai dalam waktu kurang dari 1 menit!
```bash
!python model/train.py --script sunda --epochs 30 --lr 1e-3
```

---

## 💾 Mengambil Hasil Model
Setelah training selesai, file model TFLite (`aksara_crnn.tflite`) dan file label dictionary (`labels.json`) akan tersimpan di dalam folder `model/saved/sunda/`.

Anda bisa mengunduhnya secara langsung melalui panel berkas di sebelah kiri Colab, atau jalankan perintah ini di sel baru untuk langsung mengunduh berkasnya ke PC Anda:
```python
from google.colab import files
files.download('model/saved/sunda/aksara_crnn.tflite')
files.download('model/saved/sunda/labels.json')
```
