import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

# Load dataset
files = [
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "PRSA_Data_Changping_20130301-20170228.csv",
    "PRSA_Data_Dingling_20130301-20170228.csv",
    "PRSA_Data_Dongsi_20130301-20170228.csv",
    "PRSA_Data_Guanyuan_20130301-20170228.csv"
]
locations = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan"]

# Load and preprocess datasets
dfs = {}
for loc, file in zip(locations, files):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('date', inplace=True)
    df.dropna(subset=['PM2.5'], inplace=True)
    dfs[loc] = df

# Streamlit UI
st.set_page_config(page_title="Dashboard Kualitas Udara", layout="wide")
st.title("Dashboard Analisis Kualitas Udara")

# Sidebar
with st.sidebar:
    st.header("📊 Menu Sidebar")
    
    # Pilih lokasi
    st.subheader("📍 Pilih Lokasi")
    selected_location = st.selectbox("Lokasi", locations)
    
    # Tampilkan informasi dataset
    st.subheader("📂 Informasi Dataset")
    st.write(f"Dataset yang dipilih: **{selected_location}**")
    st.write(f"Jumlah baris data: **{len(dfs[selected_location])}**")
    
    # Filter berdasarkan tanggal dengan batas waktu 2013/03/01 - 2017/02/28
    st.subheader("📅 Filter Tanggal")
    min_date = pd.Timestamp("2013-03-01")
    max_date = pd.Timestamp("2017-02-28")
    start_date = st.date_input("Tanggal Mulai", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("Tanggal Selesai", min_value=min_date, max_value=max_date, value=max_date)
    
    # Filter data berdasarkan tanggal yang dipilih
    filtered_df = dfs[selected_location].loc[start_date:end_date]
    
    # Informasi Kontak
    st.subheader("📞 Informasi Kontak")
    st.write("Nama: **Yuda Reyvandra Herman**")
    st.write("Email: **reyvandrayuda@gmail.com**")
    st.write("ID Dicoding: **MC189D5Y0450**")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Analisis Polusi Udara", "📊 Pola Polusi", "📝 Kesimpulan"])

# Tab 1: Analisis Polusi Udara
with tab1:
    st.header("📈 Analisis Polusi Udara")
    st.subheader(f"Data untuk {selected_location}")
    st.write(filtered_df.head())
    
    # Scatter plots
    scatter_plots = [('TEMP', 'PM2.5', 'Suhu vs PM2.5'),
                     ('PRES', 'PM2.5', 'Tekanan Udara vs PM2.5'),
                     ('WSPM', 'PM2.5', 'Kecepatan Angin vs PM2.5')]
    
    for x_col, y_col, title in scatter_plots:
        if x_col in filtered_df.columns and y_col in filtered_df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f'{title} - {selected_location}')
            st.pyplot(fig)
    
    # Heatmap Korelasi
    st.subheader("Matriks Korelasi Cuaca dan PM2.5")
    correlation_matrix = filtered_df[['PM2.5', 'TEMP', 'PRES', 'WSPM']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(f'Korelasi Faktor Cuaca dan PM2.5 - {selected_location}')
    st.pyplot(fig)

# Tab 2: Pola Polusi Udara
with tab2:
    st.header("📊 Pola Polusi Udara")
    st.subheader("Tren Bulanan PM2.5")
    if 'PM2.5' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))  # Perkecil ukuran gambar
        filtered_df['PM2.5'].resample('M').mean().plot(ax=ax)
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Rata-rata PM2.5')
        ax.set_title('Tren Bulanan PM2.5')
        st.pyplot(fig)
    
    # Polusi Hari Kerja vs Akhir Pekan
    st.subheader("PM2.5 di Hari Kerja vs Akhir Pekan")
    filtered_df['weekday'] = filtered_df.index.weekday
    filtered_df['is_weekend'] = filtered_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    fig, ax = plt.subplots(figsize=(6, 4))  # Perkecil ukuran gambar
    sns.barplot(x='is_weekend', y='PM2.5', data=filtered_df, estimator=np.mean, ci="sd", palette=["blue", "red"], ax=ax)
    ax.set_xticklabels(["Hari Kerja", "Akhir Pekan"])
    ax.set_title(f"Rata-rata PM2.5 di Hari Kerja vs Akhir Pekan - {selected_location}")
    st.pyplot(fig)

# Tab 3: Kesimpulan
with tab3:
    st.header("📝 Kesimpulan")
    st.subheader("Analisis Polusi Udara")
    st.write("Data menunjukkan hubungan antara faktor cuaca dan polusi PM2.5. Berikut ringkasan statistik untuk lokasi yang dipilih:")
    if 'PM2.5' in filtered_df.columns:
        st.table(filtered_df['PM2.5'].describe())
    
    # Rata-rata PM2.5 di hari kerja dan akhir pekan
    st.subheader("Perbandingan PM2.5 Hari Kerja vs Akhir Pekan")
    weekend_comparison = filtered_df.groupby('is_weekend')['PM2.5'].mean().reset_index()
    weekend_comparison['is_weekend'] = weekend_comparison['is_weekend'].map({0: 'Hari Kerja', 1: 'Akhir Pekan'})
    st.table(weekend_comparison)
    
    # Kesimpulan
    st.subheader("Kesimpulan Umum")
    st.write(
        "1. **Polusi udara** cenderung bervariasi tergantung faktor cuaca seperti suhu, tekanan udara, dan kecepatan angin.\n"
        "2. **Hari kerja vs akhir pekan**: Beberapa lokasi menunjukkan perbedaan signifikan dalam tingkat polusi.\n"
        "3. **Rekomendasi**: Pemantauan dan kebijakan pengurangan emisi sangat diperlukan untuk meningkatkan kualitas udara."
    )

st.markdown("---")
st.write("© 2025 Yuda Reyvandra Herman. All rights reserved.")
