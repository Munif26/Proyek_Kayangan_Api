import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import io

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA DASHBOARD
# ==========================================
st.set_page_config(
    page_title="Dashboard SVM Kayangan Api", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🔥 Sistem Analisis Sentimen Wisata Kayangan Api Bojonegoro")
st.markdown("Aplikasi berbasis Web ini mengimplementasikan algoritma **Support Vector Machine (SVM)** untuk mengklasifikasikan ulasan pengunjung dari Google Maps.")

# Inisialisasi NLP Sastrawi secara Global agar reusable
@st.cache_resource
def init_sastrawi():
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    stop_factory = StopWordRemoverFactory()
    stopword = stop_factory.create_stop_word_remover()
    return stemmer, stopword

stemmer, stopword = init_sastrawi()

# Fungsi Cleaning & Stemming Standar
def preprocessing_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# ==========================================
# 2. ENGINE PARSING WAKTU RELATIF GOOGLE MAPS
# ==========================================
def parse_relative_date(date_str):
    base_date = datetime.now()
    if pd.isna(date_str):
        return base_date
    s = str(date_str).lower().replace('edited', '').strip()
    
    digits = re.findall(r'\d+', s)
    num = int(digits[0]) if digits else 1
    
    if 'day' in s:
        return base_date - timedelta(days=num)
    elif 'week' in s:
        return base_date - timedelta(weeks=num)
    elif 'month' in s:
        return base_date - timedelta(days=num * 30)
    elif 'year' in s:
        return base_date - timedelta(days=num * 365)
    return base_date

# ==========================================
# 3. CORE DATA PROCESSING (CACHED FOR SPEED)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_excel("new dataset_kayangan_api_scrapping.xlsx")
    df = df.dropna(subset=['isi_komentar'])
    df = df.drop_duplicates(subset=['isi_komentar']).reset_index(drop=True)
    
    # Fungsi Pelabelan Lexicon
    def label_sentimen(text):
        positif = ['indah','bagus','keren','sejuk','nyaman','mantap','recommended','suka','terbaik','cocok','baik','ramah','puas','seru','bersih','enak','menarik','unik']
        negatif = ['macet','kotor','mahal','rusak','kecewa','buruk','panas','antri','capek','jelek','kurang','parah','susah','bau','lama','lelah','sempit']
        words = text.split()
        skor_pos = sum(word in positif for word in words)
        skor_neg = sum(word in negatif for word in words)
        return 'positif' if skor_pos >= skor_neg else 'negatif'
    
    df['clean_text'] = df['isi_komentar'].apply(preprocessing_text)
    df['label'] = df['clean_text'].apply(label_sentimen)
    df['parsed_date'] = df['tanggal_ulasan'].apply(parse_relative_date)
    return df

df_clean = load_and_preprocess_data()

# ==========================================
# 4. TRAINING MODEL SVM (CACHED RESOURCE)
# ==========================================
@st.cache_resource
def train_svm_model(data):
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(data['clean_text'])
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return tfidf, model, acc, cm, y_test, y_pred

tfidf_vectorizer, svm_model, accuracy, conf_matrix, y_test, y_pred = train_svm_model(df_clean)

# ==========================================
# 5. KONTROL INTERAKTIF (SIDEBAR FILTER)
# ==========================================
st.sidebar.header("🗓️ Filter Rentang Waktu")
min_date = df_clean['parsed_date'].min().date()
max_date = df_clean['parsed_date'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Tentukan Durasi Analisis:",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

mask = (df_clean['parsed_date'].dt.date >= start_date) & (df_clean['parsed_date'].dt.date <= end_date)
df_filtered = df_clean.loc[mask]

# ==========================================
# 6. PENYUSUNAN ANTARMUKA UTAMA (TABS UI)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Statistik & Tren Objek Wisata", "🧠 Performa Model SVM", "🔮 Live Classifier & Upload Dataset"])

# --- TAB 1: METRIK UTAMA & GRAFIK TREN ---
with tab1:
    st.subheader("📋 Ringkasan Sentimen Wisatawan")
    total_data = len(df_filtered)
    
    if total_data > 0:
        pos_count = len(df_filtered[df_filtered['label'] == 'positif'])
        neg_count = len(df_filtered[df_filtered['label'] == 'negatif'])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Ulasan Terfilter", f"{total_data} Data")
        c2.metric("Sentimen Positif (Puas)", f"{(pos_count/total_data)*100:.1f}%", f"{pos_count} Ulasan")
        c3.metric("Sentimen Negatif (Komplain)", f"{(neg_count/total_data)*100:.1f}%", f"-{neg_count} Keluhan", delta_color="inverse")
        
        st.markdown("---")
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            st.write("**Persentase Kepuasan (Pie Chart)**")
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            df_filtered['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=140, ax=ax1)
            ax1.set_ylabel('')
            st.pyplot(fig1)
            
        with col_bar:
            st.write("**Visualisasi Sebaran Kata Utama (Word Cloud)**")
            text_all = ' '.join(df_filtered['clean_text'])
            if text_all.strip():
                wc = WordCloud(width=500, height=350, background_color='white', colormap='plasma').generate(text_all)
                fig2, ax2 = plt.subplots()
                ax2.imshow(wc, interpolation='bilinear')
                ax2.axis('off')
                st.pyplot(fig2)
                
        st.markdown("---")
        st.write("**📈 Tren Perubahan Sentimen Wisatawan**")
        opsi_waktu = st.radio("Metode Pengelompokan Waktu:", ('Harian', 'Bulanan', 'Tahunan'), horizontal=True)
        
        df_trend = df_filtered.copy()
        if opsi_waktu == 'Harian':
            df_trend['Periode'] = df_trend['parsed_date'].dt.to_period('D')
        elif opsi_waktu == 'Bulanan':
            df_trend['Periode'] = df_trend['parsed_date'].dt.to_period('M')
        else:
            df_trend['Periode'] = df_trend['parsed_date'].dt.to_period('Y')
            
        trend_res = df_trend.groupby(['Periode', 'label']).size().unstack(fill_value=0)
        trend_res.index = trend_res.index.astype(str)
        
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        if 'positif' in trend_res:
            ax3.plot(trend_res.index, trend_res['positif'], color='#2ecc71', marker='o', label='Positif')
        if 'negatif' in trend_res:
            ax3.plot(trend_res.index, trend_res['negatif'], color='#e74c3c', marker='s', label='Negatif')
        plt.xticks(rotation=45)
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.warning("Data tidak ditemukan pada filter rentang waktu tersebut.")

# --- TAB 2: EVALUASI AKADEMIK SVM ---
with tab2:
    st.subheader("⚡ Validasi Saintifik Algoritma SVM")
    
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.metric("Akurasi Pengujian Model (Real)", f"{accuracy*100:.2f}%")
        st.write("Hasil evaluasi klasifikasi menggunakan pembagian data 80:20.")
        
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['negatif', 'positif'], yticklabels=['negatif', 'positif'], ax=ax_cm)
        ax_cm.set_xlabel('Prediksi Model')
        ax_cm.set_ylabel('Aktual Lapangan')
        st.pyplot(fig_cm)
        
    with col_m2:
        st.write("**Classification Report (Tabel Evaluasi Dosen):**")
        rep_dict = classification_report(y_test, y_pred, output_dict=True)
        df_rep = pd.DataFrame(rep_dict).transpose()
        st.dataframe(df_rep.style.format(precision=2))
        
        st.write("**10 Kata Kunci Paling Berpengaruh pada Hyperplane SVM:**")
        coef = svm_model.coef_.toarray()[0]
        features = tfidf_vectorizer.get_feature_names_out()
        df_feat = pd.DataFrame({'kata': features, 'bobot_pengaruh': np.abs(coef)})
        df_feat = df_feat.sort_values(by='bobot_pengaruh', ascending=False).head(10)
        
        fig_feat, ax_feat = plt.subplots(figsize=(7, 3))
        sns.barplot(x='bobot_pengaruh', y='kata', data=df_feat, palette='viridis', ax=ax_feat)
        st.pyplot(fig_feat)

# --- TAB 3: LIVE CLASSIFIER & BULK UPLOAD (REVISI DOSEN) ---
with tab3:
    st.subheader("🔮 Pengujian Kepuasan Ulasan Real-Time & Unggah Massal")
    
    # Membuat sub-pilihan menu menggunakan selectbox
    pilihan_input = st.selectbox(
        "Pilih Metode Analisis:",
        ["Opsi 1: Input Teks Manual (Satu Per Satu)", "Opsi 2: Upload File Excel/CSV (Prediksi Massal)"]
    )
    
    # -----------------------------------------------------
    # MIKRO-FITUR 1: INPUT MANUAL (FITUR LAMA)
    # -----------------------------------------------------
    if "Opsi 1" in pilihan_input:
        st.write("Ketik ulasan tunggal di bawah ini untuk dideteksi oleh Kecerdasan Buatan SVM.")
        input_text = st.text_area("Tulis Ulasan Pengunjung:", placeholder="Contoh: Tempatnya bagus bgt, tapi sayang jalannya kalau malam masih gelap bgt...")
        
        if st.button("Analisis Teks"):
            if input_text.strip():
                # Proses preprocessing ulasan baru
                clean_input = preprocessing_text(input_text)
                vec_input = tfidf_vectorizer.transform([clean_input])
                prediksi = svm_model.predict(vec_input)[0]
                
                st.markdown("### **Hasil Analisis Teks:**")
                if prediksi == 'positif':
                    st.success("🎉 **SENTIMEN POSITIF** — Pengunjung merasa puas dengan destinasi wisata Kayangan Api.")
                else:
                    st.error("🚨 **SENTIMEN NEGATIF** — Ulasan mengandung kritik atau saran perbaikan fasilitas.")
            else:
                st.warning("Harap ketik ulasan terlebih dahulu.")

    # -----------------------------------------------------
    # MIKRO-FITUR 2: BULK UPLOAD EXCEL/CSV (FITUR BARU UNTUK DOSEN)
    # -----------------------------------------------------
    else:
        st.write("Unggah file Excel (`.xlsx`) atau CSV hasil scraping baru. Sistem akan mendeteksi seluruh ulasan secara otomatis.")
        st.info("⚠️ **Format Penting:** File yang di-upload WAJIB memiliki kolom bernama **`isi_komentar`** agar sistem bisa membaca teks ulasannya.")
        
        uploaded_file = st.file_uploader("Pilih File Ulasan Terbaru:", type=["xlsx", "csv"])
        
        if uploaded_file is not None:
            # Membaca jenis file yang diunggah
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    df_uploaded = pd.read_excel(uploaded_file)
                else:
                    df_uploaded = pd.read_csv(uploaded_file)
                
                # Cek ketersediaan kolom target
                if 'isi_komentar' in df_uploaded.columns:
                    st.success(f"Berhasil memuat data! Menemukan {len(df_uploaded)} ulasan baru.")
                    
                    # Jalankan animasi loading agar interaktif
                    with st.spinner('Model SVM sedang mengklasifikasikan data massal...'):
                        # 1. Jalankan Preprocessing Massal
                        df_uploaded['clean_text_temp'] = df_uploaded['isi_komentar'].apply(preprocessing_text)
                        
                        # 2. Transformasi Vektor TF-IDF Massal
                        X_uploaded = tfidf_vectorizer.transform(df_uploaded['clean_text_temp'])
                        
                        # 3. Prediksi Massal menggunakan Model SVM yang Sudah Terlatih
                        df_uploaded['Prediksi_Sentimen_SVM'] = svm_model.predict(X_uploaded)
                        
                        # Hapus kolom sementara agar file bersih saat di-download
                        df_uploaded = df_uploaded.drop(columns=['clean_text_temp'])
                    
                    st.balloons() # Efek perayaan fungsional sukses
                    
                    # Tampilkan Statistik Hasil Upload Baru
                    st.markdown("### 📊 Ringkasan Hasil Klasifikasi Massal")
                    res_upload = df_uploaded['Prediksi_Sentimen_SVM'].value_counts()
                    
                    col_u1, col_u2 = st.columns(2)
                    with col_u1:
                        # Grafik Sebaran Hasil Upload Baru
                        fig_up, ax_up = plt.subplots(figsize=(6, 4))
                        sns.barplot(x=res_upload.index, y=res_upload.values, palette='Set2', ax=ax_up, hue=res_upload.index, legend=False)
                        ax_up.set_ylabel("Jumlah")
                        ax_up.set_xlabel("Kategori Sentimen")
                        st.pyplot(fig_up)
                    
                    with col_u2:
                        st.write("**Pratinjau Data Hasil Prediksi Model:**")
                        st.dataframe(df_uploaded[['isi_komentar', 'Prediksi_Sentimen_SVM']].head(10))
                    
                    # DOWNLOAD BUTTON UNTUK EXPORT FILE BARU
                    st.markdown("---")
                    st.write("**💾 Ambil File Hasil Analisis**")
                    st.write("Kamu bisa mendownload kembali file yang kamu unggah yang kini sudah lengkap dengan kolom sentimen hasil tebakan AI SVM.")
                    
                    # Konversi DataFrame ke format Excel di dalam memori buffer
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_uploaded.to_excel(writer, index=False, sheet_name='Hasil_Sentimen_SVM')
                    
                    st.download_button(
                        label="📥 Download Hasil Analisis Sentimen (Excel)",
                        data=buffer.getvalue(),
                        file_name="hasil_analisis_sentimen_kayangan_api.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error("Gagal memproses! Kolom 'isi_komentar' tidak ditemukan pada file yang diunggah. Silakan periksa kembali nama kolom file Anda.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membaca file: {e}")