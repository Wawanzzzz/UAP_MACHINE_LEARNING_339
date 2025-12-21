# Klasifikasi Outfit Biasa dan Skena

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem klasifikasi outfit menjadi dua kelas, yaitu
**biasa** dan **skena**, menggunakan pendekatan pembelajaran mesin berbasis data citra.

## Dataset
Dataset terdiri dari lebih dari 6.000 citra outfit yang terbagi ke dalam dua kelas:
- Biasa
- Skena

Dataset dibagi menjadi data latih, validasi, dan uji menggunakan rasio 70:15:15.
Sumber dataset: (isi sumber kamu, misalnya koleksi pribadi / Kaggle / dsb).

## Preprocessing
- Resize citra menjadi 224x224
- Normalisasi menggunakan mean dan std ImageNet
- Split data secara terstruktur ke dalam folder train, val, dan test

## Model yang Digunakan
1. CNN (Non-Pretrained)
2. MobileNetV2 (Transfer Learning)
3. EfficientNet-B0 (Transfer Learning)

## Hasil Evaluasi dan Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| CNN | 0.94 | 0.95 | 0.93 | 0.94 |
| MobileNetV2 | 0.93 | 0.93 | 0.93 | 0.93 |
| EfficientNet-B0 | 0.96 | 0.96 | 0.96 | 0.96 |

## Analisis
EfficientNet-B0 menunjukkan performa terbaik dengan akurasi tertinggi dan keseimbangan
precision serta recall yang baik dibandingkan model lainnya.

## Implementasi Website
Sistem website sederhana dibangun menggunakan Streamlit untuk melakukan klasifikasi
outfit secara interaktif. Pengguna dapat mengunggah gambar dan memilih model untuk
melihat hasil prediksi.

## Cara Menjalankan Aplikasi
```bash
python -m pdm run streamlit run src/app.py
