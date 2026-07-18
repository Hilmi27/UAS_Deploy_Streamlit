# ❤️ Heart Disease Risk Prediction — UAS Pembelajaran Mesin

Aplikasi web berbasis Machine Learning untuk memprediksi risiko penyakit jantung, dibangun sebagai Capstone Project UAS Mata Kuliah Pembelajaran Mesin.

## 🔗 Link

- **Live App:** _[(isi setelah deploy ke Streamlit Community Cloud)](https://uas-ml-hilmi-riyan.streamlit.app/)_
- **Dataset:** [Kaggle - Heart Disease Risk Prediction Dataset](https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset)

## 📁 Struktur Repository

```
├── app.py                  # Aplikasi Streamlit utama
├── requirements.txt        # Daftar dependency Python
├── model/
│   ├── rf_model.pkl         # Model Random Forest (terbaik) - joblib
│   ├── imputer.pkl          # SimpleImputer terlatih
│   ├── scaler.pkl           # StandardScaler terlatih
│   ├── feature_columns.pkl  # Urutan kolom fitur
│   └── metrics.json         # Hasil evaluasi (accuracy, precision, recall, F1, confusion matrix)
├── data/
│   └── heart_disease_risk_dataset_earlymed.csv
└── notebook/
    └── uasML_fixed.ipynb    # Notebook lengkap: EDA, preprocessing, modeling, evaluasi
```

## 🎯 Problem Statement

Memprediksi risiko penyakit jantung individu berdasarkan kombinasi gejala klinis dan faktor risiko gaya hidup, sebagai alat bantu skrining awal (early screening tool). Detail lengkap ada di notebook (`notebook/uasML_fixed.ipynb`), bagian Soal 1.

## 🧪 Metodologi

1. **EDA & Preprocessing** — cek missing values/duplikat/inconsistency, analisis univariat & multivariat, imputasi & scaling (fit hanya di data train), feature selection.
2. **Modeling** — 3 model dibandingkan: Decision Tree, Random Forest, KNN.
3. **Evaluasi** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix pada data test.
4. **Model Terbaik** — Random Forest (F1-Score ≈ 99.15%).
5. **Deployment** — Aplikasi Streamlit dengan 5 halaman: Dashboard EDA, Model Demo, Evaluasi Model, Interpretasi Hasil, Dokumentasi.

## 🚀 Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Disclaimer

Dataset yang digunakan bersifat sintetis untuk keperluan akademik. Prediksi aplikasi ini bukan diagnosis medis.

## 👥 Kelompok

- Riyan Andriansyah - A11.2024.15783
- Hilmi Putra Dwi Suryono Muhammad - A11.2025.16591
- Kelompok A11.4407 — Pembelajaran Mesin, UDINUS
