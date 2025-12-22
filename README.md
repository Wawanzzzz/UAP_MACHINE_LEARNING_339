# Klasifikasi Outfit Biasa dan Skena Menggunakan Deep Learning

## Deskripsi Proyek
Proyek ini merupakan implementasi pembelajaran mesin berbasis data citra untuk melakukan
klasifikasi outfit ke dalam dua kelas, yaitu **biasa** dan **skena**.  
Tujuan utama proyek ini adalah untuk menerapkan konsep *Neural Network* dasar,
*Transfer Learning*, serta integrasi model ke dalam sistem website sederhana menggunakan
**Streamlit**, sesuai dengan materi Modul Pembelajaran Mesin 1–6.

Proyek ini dikembangkan sebagai bagian dari **Ujian Akhir Praktikum (UAP)** mata kuliah
Pembelajaran Mesin.

---

## Dataset
Dataset yang digunakan berupa data citra outfit dengan dua kelas:
- **Biasa**
- **Skena**

Jumlah total data lebih dari **6.000 citra**, dengan distribusi yang relatif seimbang:
- Biasa: ±3.200 citra  
- Skena: ±3.000 citra  

Dataset disimpan dalam format gambar (`.jpg`, `.jpeg`, `.png`) dan dibagi ke dalam tiga bagian:
- Data latih (train)
- Data validasi (validation)
- Data uji (test)

Pembagian dataset dilakukan secara terstruktur dengan rasio **70% : 15% : 15%**.

**Sumber Dataset:**  
Dataset dikumpulkan dari sumber terbuka dan koleksi pribadi, kemudian diseleksi dan
disesuaikan dengan kebutuhan klasifikasi outfit.

---

## Preprocessing Data
Tahapan preprocessing yang dilakukan pada data citra meliputi:
1. Resize citra menjadi ukuran **224 × 224**
2. Konversi citra ke tensor
3. Normalisasi menggunakan mean dan standar deviasi ImageNet
4. Pembagian dataset ke dalam folder `train`, `val`, dan `test`

Tahapan preprocessing ini disesuaikan agar kompatibel dengan model CNN dan model pretrained.

---

## Model yang Digunakan
Pada proyek ini diimplementasikan **tiga model pembelajaran mesin**, terdiri dari:

### 1. CNN (Non-Pretrained)
Model Convolutional Neural Network yang dibangun dan dilatih dari awal tanpa menggunakan
bobot pretrained. Model ini digunakan sebagai **baseline** untuk melihat kemampuan model
sederhana dalam mengklasifikasikan outfit.

### 2. MobileNetV2 (Transfer Learning)
Model pretrained MobileNetV2 digunakan dengan pendekatan *transfer learning*.
Layer feature extractor dibekukan (*freeze*), dan hanya layer classifier yang dilatih ulang
untuk menyesuaikan dengan jumlah kelas.

### 3. EfficientNet-B0 (Transfer Learning)
EfficientNet-B0 digunakan sebagai model pretrained kedua. Model ini memiliki arsitektur yang
lebih optimal dalam menyeimbangkan kompleksitas dan performa, sehingga diharapkan memberikan
hasil klasifikasi yang lebih baik.

---

## Evaluasi Model
Evaluasi performa model dilakukan menggunakan data uji (*test set*) dengan metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

### Tabel Perbandingan Performa Model

| Model | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| CNN (Non-Pretrained) | 0.94 | 0.95 | 0.93 | 0.94 |
| MobileNetV2 (Pretrained) | 0.93 | 0.93 | 0.93 | 0.93 |
| EfficientNet-B0 (Pretrained) | **0.96** | **0.96** | **0.96** | **0.96** |

---

## Analisis Perbandingan Model
Berdasarkan hasil evaluasi, model CNN non-pretrained mampu memberikan performa yang cukup baik
meskipun dilatih dari awal. Hal ini menunjukkan bahwa arsitektur CNN dasar sudah mampu
menangkap pola visual pada outfit.

MobileNetV2 memberikan performa yang stabil dengan ukuran model yang relatif ringan, sehingga
cocok digunakan pada sistem dengan keterbatasan sumber daya.

EfficientNet-B0 menunjukkan performa terbaik di antara ketiga model, dengan nilai akurasi,
precision, recall, dan F1-score yang paling tinggi dan seimbang. Hal ini disebabkan oleh
arsitektur EfficientNet yang lebih efisien dalam memanfaatkan parameter model.

---

## Implementasi Sistem Website (Streamlit)
Sistem website sederhana dibangun menggunakan **Streamlit** untuk mendemonstrasikan hasil
klasifikasi outfit secara interaktif.

Fitur utama aplikasi:
- Upload gambar outfit oleh pengguna
- Pemilihan model klasifikasi (CNN, MobileNetV2, EfficientNet-B0)
- Menampilkan hasil prediksi kelas
- Menampilkan nilai confidence (probabilitas)

Aplikasi dijalankan secara **lokal** pada komputer praktikan.

---

## Cara Menjalankan Aplikasi
Pastikan seluruh dependensi telah terpasang menggunakan PDM, kemudian jalankan perintah berikut
dari root project:

```bash
python -m pdm run streamlit run src/app.py

---

## Note
Dikarenakan terdapat kendala pada vs code saya, sehingga tidak bisa push github dengan baik, jadi saya melakukan push secara manual dengan cara upload file satu persatu. Dibawah ini adalah link Drive dari dataset saya:

```bash
https://drive.google.com/drive/folders/1dNgbLvTf1p8bEUbVHHd8JlWyBJKnWn7S?usp=sharing
