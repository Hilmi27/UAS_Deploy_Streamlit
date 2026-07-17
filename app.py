import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# ============================================================
# LOAD ARTIFACTS (model, preprocessing, metrics, data)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/rf_model.pkl")
    imputer = joblib.load("model/imputer.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    with open("model/metrics.json") as f:
        metrics = json.load(f)
    return model, imputer, scaler, feature_columns, metrics

@st.cache_data
def load_data():
    df = pd.read_csv("data/heart_disease_risk_dataset_earlymed.csv")
    df = df.drop_duplicates()
    return df

model, imputer, scaler, feature_columns, metrics = load_artifacts()
df = load_data()

SYMPTOM_FEATURES = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
    "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea"
]
RISK_FACTOR_FEATURES = [
    "High_BP", "High_Cholesterol", "Diabetes", "Smoking",
    "Obesity", "Sedentary_Lifestyle", "Family_History", "Chronic_Stress"
]

LABELS = {
    "Chest_Pain": "Nyeri Dada",
    "Shortness_of_Breath": "Sesak Napas",
    "Fatigue": "Mudah Lelah",
    "Palpitations": "Jantung Berdebar",
    "Dizziness": "Pusing",
    "Swelling": "Pembengkakan (Edema)",
    "Pain_Arms_Jaw_Back": "Nyeri Lengan/Rahang/Punggung",
    "Cold_Sweats_Nausea": "Keringat Dingin & Mual",
    "High_BP": "Tekanan Darah Tinggi",
    "High_Cholesterol": "Kolesterol Tinggi",
    "Diabetes": "Diabetes",
    "Smoking": "Merokok",
    "Obesity": "Obesitas",
    "Sedentary_Lifestyle": "Gaya Hidup Kurang Gerak",
    "Family_History": "Riwayat Keluarga",
    "Chronic_Stress": "Stres Kronis",
    "Gender": "Jenis Kelamin (1 = Laki-laki, 0 = Perempuan)",
}

# ============================================================
# SIDEBAR NAVIGASI
# ============================================================
st.sidebar.title("❤️ Heart Disease Risk App")
page = st.sidebar.radio(
    "Navigasi",
    ["Dashboard EDA", "Model Demo", "Evaluasi Model", "Interpretasi Hasil", "Dokumentasi"]
)

# ============================================================
# 1. DASHBOARD EDA
# ============================================================
if page == "Dashboard EDA":
    st.title("📊 Dashboard EDA — Heart Disease Risk Dataset")
    st.caption("Eksplorasi interaktif dari dataset yang digunakan untuk melatih model.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{df.shape[0]:,}")
    col2.metric("Jumlah Fitur", f"{df.shape[1] - 1}")
    col3.metric("Risiko Tinggi", f"{(df['Heart_Risk']==1).sum():,}")
    col4.metric("Risiko Rendah", f"{(df['Heart_Risk']==0).sum():,}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Distribusi Target", "Distribusi Usia", "Korelasi Fitur"])

    with tab1:
        fig, ax = plt.subplots(figsize=(5, 4))
        df["Heart_Risk"].value_counts().rename({0: "Risiko Rendah", 1: "Risiko Tinggi"}).plot(
            kind="bar", color=["#4C72B0", "#C44E52"], ax=ax
        )
        ax.set_ylabel("Jumlah")
        ax.set_title("Distribusi Heart Risk")
        st.pyplot(fig)
        st.info("Dataset seimbang sempurna (50%/50%) antara kelas risiko tinggi dan rendah.")

    with tab2:
        age_filter = st.slider("Filter rentang usia", int(df.Age.min()), int(df.Age.max()),
                                (int(df.Age.min()), int(df.Age.max())))
        filtered = df[(df.Age >= age_filter[0]) & (df.Age <= age_filter[1])]
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=filtered, x="Age", hue="Heart_Risk", multiple="stack",
                     palette=["#4C72B0", "#C44E52"], ax=ax)
        ax.set_title("Distribusi Usia berdasarkan Heart Risk")
        st.pyplot(fig)
        st.info(f"Rata-rata usia kelompok risiko tinggi: **{df[df.Heart_Risk==1].Age.mean():.1f} tahun** "
                f"vs kelompok risiko rendah: **{df[df.Heart_Risk==0].Age.mean():.1f} tahun**.")

    with tab3:
        selected_features = st.multiselect(
            "Pilih fitur untuk ditampilkan di heatmap",
            options=list(df.columns),
            default=["Age", "High_BP", "Diabetes", "Smoking", "Obesity", "Chest_Pain", "Heart_Risk"]
        )
        if len(selected_features) >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[selected_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Pilih minimal 2 fitur.")

# ============================================================
# 2. MODEL DEMO
# ============================================================
elif page == "Model Demo":
    st.title("🩺 Model Demo — Prediksi Risiko Penyakit Jantung")
    st.caption("Masukkan data gejala dan faktor risiko untuk mendapatkan prediksi.")

    with st.form("prediction_form"):
        st.subheader("Data Diri")
        c1, c2 = st.columns(2)
        age = c1.number_input("Usia", min_value=1, max_value=120, value=45)
        gender = c2.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])
        gender_val = 1 if gender == "Laki-laki" else 0

        st.subheader("Gejala Klinis")
        symptom_vals = {}
        cols = st.columns(4)
        for i, feat in enumerate(SYMPTOM_FEATURES):
            symptom_vals[feat] = 1 if cols[i % 4].checkbox(LABELS[feat], key=feat) else 0

        st.subheader("Faktor Risiko Gaya Hidup & Riwayat")
        risk_vals = {}
        cols2 = st.columns(4)
        for i, feat in enumerate(RISK_FACTOR_FEATURES):
            risk_vals[feat] = 1 if cols2[i % 4].checkbox(LABELS[feat], key=feat) else 0

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
        input_dict = {**symptom_vals, **risk_vals, "Gender": gender_val, "Age": age}
        input_df = pd.DataFrame([input_dict])[feature_columns]

        input_df = pd.DataFrame(imputer.transform(input_df), columns=feature_columns)
        input_df["Age"] = scaler.transform(input_df[["Age"]])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ **Hasil: RISIKO TINGGI** penyakit jantung (probabilitas {proba[1]*100:.1f}%)")
            st.warning("Disarankan untuk berkonsultasi dengan tenaga medis untuk pemeriksaan lebih lanjut. "
                       "Hasil ini adalah alat bantu skrining awal, bukan diagnosis medis.")
        else:
            st.success(f"✅ **Hasil: RISIKO RENDAH** penyakit jantung (probabilitas {proba[0]*100:.1f}%)")
            st.info("Tetap jaga pola hidup sehat. Hasil ini adalah alat bantu skrining awal, bukan diagnosis medis.")

        fig, ax = plt.subplots(figsize=(5, 1.2))
        ax.barh([0], [proba[1]], color="#C44E52")
        ax.barh([0], [proba[0]], left=[proba[1]], color="#4C72B0")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Probabilitas")
        st.pyplot(fig)

# ============================================================
# 3. EVALUASI MODEL
# ============================================================
elif page == "Evaluasi Model":
    st.title("📈 Evaluasi Model")
    st.caption("Perbandingan performa 3 model yang telah dilatih pada data uji (test set).")

    results_df = pd.DataFrame(metrics["results_table"]).sort_values(by="F1 Score", ascending=False)
    st.dataframe(
        results_df.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                                  "Recall": "{:.4f}", "F1 Score": "{:.4f}"})
        .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="lightgreen"),
        use_container_width=True
    )

    st.divider()
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Pilih model", options=list(metrics["confusion_matrices"].keys()))
    cm = np.array(metrics["confusion_matrices"][selected_model])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rendah", "Tinggi"], yticklabels=["Rendah", "Tinggi"], ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

    st.divider()
    st.subheader("Classification Report")
    report = metrics["classification_reports"][selected_model]
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    st.info(f"📌 Data uji terdiri dari **{metrics['n_test']:,}** sampel "
            f"(dari total split: {metrics['n_train']:,} train / {metrics['n_val']:,} validation / {metrics['n_test']:,} test).")

# ============================================================
# 4. INTERPRETASI HASIL
# ============================================================
elif page == "Interpretasi Hasil":
    st.title("🔍 Interpretasi Hasil & Insight Bisnis")

    st.subheader(f"Model Terbaik: {metrics['best_model']}")
    st.markdown("""
    **Random Forest** dipilih sebagai model terbaik karena unggul di seluruh metrik, khususnya
    **Recall** — metrik paling krusial dalam konteks skrining medis, karena kesalahan
    *false negative* (pasien berisiko tinggi diprediksi rendah risiko) jauh lebih berbahaya
    dibanding *false positive*.
    """)

    if metrics.get("feature_importance"):
        st.subheader("Feature Importance")
        fi = pd.Series(metrics["feature_importance"]).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        fi.plot(kind="barh", ax=ax, color="#C44E52")
        ax.set_xlabel("Importance")
        st.pyplot(fig)

        top_feat = fi.sort_values(ascending=False).index[0]
        st.info(f"Fitur paling berpengaruh terhadap prediksi model adalah **{LABELS.get(top_feat, top_feat)}**.")

    st.subheader("Insight Bisnis")
    st.markdown("""
    - Usia dan gejala klinis (nyeri dada, sesak napas, keringat dingin) adalah prediktor terkuat risiko jantung.
    - Faktor gaya hidup (merokok, obesitas, diabetes, tekanan darah tinggi) tetap berkontribusi signifikan,
      meski pengaruhnya sedikit lebih rendah dibanding gejala klinis langsung.
    - Model ini cocok digunakan sebagai **alat bantu skrining awal**, bukan pengganti diagnosis medis profesional.
    """)

    st.warning("""
    ⚠️ **Catatan Limitasi:** Dataset yang digunakan bersifat sintetis (bukan data rekam medis riil),
    ditandai dengan distribusi kelas yang seimbang sempurna dan performa model yang sangat tinggi (>97%
    di semua model). Performa pada data klinis riil kemungkinan akan lebih rendah karena data riil
    memiliki lebih banyak noise dan faktor perancu.
    """)

# ============================================================
# 5. DOKUMENTASI
# ============================================================
elif page == "Dokumentasi":
    st.title("📄 Dokumentasi")

    st.subheader("Tentang Dataset")
    st.markdown("""
    - **Nama:** Heart Disease Risk Prediction Dataset
    - **Sumber:** [Kaggle - mahatiratusher](https://www.kaggle.com/datasets/mahatiratusher/heart-disease-risk-prediction-dataset)
    - **Jumlah data:** 70.000 baris (63.755 setelah penghapusan duplikat), 18 fitur + 1 target
    - **Fitur:** 17 fitur biner (gejala klinis & faktor risiko) + 1 fitur numerik (Age)
    - **Target:** `Heart_Risk` (0 = Risiko Rendah, 1 = Risiko Tinggi)
    """)

    st.subheader("Metodologi")
    st.markdown("""
    1. **Problem Definition** — Klasifikasi biner risiko penyakit jantung untuk alat bantu skrining awal.
    2. **EDA & Preprocessing** — Cek missing values, duplikat, inconsistency, outlier; analisis univariat/multivariat;
       imputasi & scaling (fit hanya di data train untuk mencegah data leakage); feature selection (SelectKBest).
    3. **Modeling** — 3 model dibandingkan: Decision Tree, Random Forest, KNN. Evaluasi dengan Accuracy, Precision,
       Recall, F1-Score, dan Confusion Matrix pada data test.
    4. **Deployment** — Model terbaik (Random Forest) diserialisasi dengan `joblib` dan digunakan di aplikasi
       Streamlit ini untuk prediksi real-time.
    """)

    st.subheader("Cara Penggunaan Aplikasi")
    st.markdown("""
    - **Dashboard EDA**: eksplorasi interaktif dataset (distribusi, korelasi).
    - **Model Demo**: isi form gejala & faktor risiko, klik "Prediksi Sekarang" untuk melihat hasil prediksi.
    - **Evaluasi Model**: bandingkan performa 3 model, lihat confusion matrix & classification report.
    - **Interpretasi Hasil**: pahami fitur mana yang paling berpengaruh dan insight bisnisnya.
    """)

    st.subheader("Disclaimer")
    st.error("""
    Aplikasi ini dibuat untuk **tujuan akademik (UAS Pembelajaran Mesin)** dan menggunakan dataset sintetis.
    Hasil prediksi **bukan diagnosis medis** dan tidak boleh dijadikan dasar keputusan medis tanpa
    konsultasi dengan tenaga profesional.
    """)
