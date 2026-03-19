import streamlit as st
import requests
import pandas as pd
import io

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Home Credit Score Card",
    page_icon="💳",
    layout="wide"
)

# 2. Judul & Header
st.title("💳 Home Credit Risk Analysis")
st.markdown("Pilih metode analisis: Input Individu atau Unggah Batch (CSV).")

# --- 3. NEW: TABS SYSTEM ---
tab_single, tab_batch = st.tabs(["👤 Analisis Individu", "📂 Analisis Batch (CSV)"])

with tab_single:
    st.header("👤 Analisis Risiko Kredit Individu")
    st.markdown("Masukkan data nasabah untuk mendapatkan analisis risiko secara real-time.") 
    # 3. Form Input Data Nasabah
    # Kita bagi menjadi 2 kolom agar tidak terlalu panjang ke bawah
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Skor Eksternal & Rating")
        ext_1 = st.slider("EXT_SOURCE_1 (Skor Kredit 1)", 0.0, 1.0, 0.5)
        ext_2 = st.slider("EXT_SOURCE_2 (Skor Kredit 2)", 0.0, 1.0, 0.5)
        ext_3 = st.slider("EXT_SOURCE_3 (Skor Kredit 3)", 0.0, 1.0, 0.5)
        region_rating = st.selectbox("Rating Wilayah (City)", [1, 2, 3], index=1)

        st.subheader("🎓 Profil Nasabah")
        education = st.selectbox("Tingkat Pendidikan", [
            "Higher education", 
            "Secondary / secondary special", 
            "Incomplete higher", 
            "Lower secondary", 
            "Academic degree"
        ])
        income_type = st.selectbox("Tipe Pendapatan", [
            "Working", 
            "Commercial associate", 
            "Pensioner", 
            "State servant", 
            "Student", 
            "Unemployed"
        ])

    with col2:
        st.subheader("💰 Data Finansial")
        amt_credit = st.number_input("Total Pinjaman (AMT_CREDIT)", min_value=0.0, value=500000.0)
        amt_annuity = st.number_input("Cicilan Tahunan (AMT_ANNUITY)", min_value=0.0, value=25000.0)
        amt_income = st.number_input("Pendapatan Tahunan (AMT_INCOME)", min_value=0.0, value=150000.0)
        amt_goods = st.number_input("Harga Barang (AMT_GOODS_PRICE)", min_value=0.0, value=450000.0)

        st.subheader("⏳ Riwayat & Usia")
        age_years = st.number_input("Usia (Tahun)", min_value=18, max_value=100, value=30)
        days_birth = age_years * -365 # Konversi ke format dataset
        
        emp_years = st.number_input("Lama Bekerja (Tahun)", min_value=0, max_value=60, value=5)
        days_employed = emp_years * -365 # Konversi ke format dataset

    # 4. Logika Tombol Prediksi
    if st.button("🔍 Analisis Risiko Kredit", use_container_width=True):
        # Susun payload sesuai dengan Pydantic Schema di FastAPI
        payload = {
            "EXT_SOURCE_1": ext_1,
            "EXT_SOURCE_2": ext_2,
            "EXT_SOURCE_3": ext_3,
            "AMT_ANNUITY": amt_annuity,
            "AMT_CREDIT": amt_credit,
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_GOODS_PRICE": amt_goods,
            "DAYS_BIRTH": int(days_birth),
            "DAYS_EMPLOYED": int(days_employed),
            "REGION_RATING_CLIENT_W_CITY": region_rating,
            "NAME_EDUCATION_TYPE": education,
            "NAME_INCOME_TYPE": income_type
        }

        try:
            # Panggil API FastAPI (localhost karena satu container Docker)
            with st.spinner('Sedang menghitung skor risiko...'):
                response = requests.post("http://localhost:8000/predict", json=payload)
                result = response.json()

            if response.status_code == 200:
                st.success("✅ Analisis Selesai!")
                
                # Tampilan Hasil
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Probabilitas Gagal Bayar", f"{result['probability'] * 100:.2f} %")
                    
                    # Warna Alert berdasarkan Rating
                    rating = result['risk_rating']
                    if "Low" in rating:
                        st.info(f"Kategori: {rating}")
                    elif "Medium" in rating:
                        st.warning(f"Kategori: {rating}")
                    else:
                        st.error(f"Kategori: {rating}")
                
                with res_col2:
                    st.write("**System Analysis:**")
                    # Gunakan markdown agar bold dan emoji muncul dengan benar
                    st.markdown(result['message'])
                    
                # Progress Bar sebagai "Risk Meter"
                st.progress(result['probability'])
                
            else:
                st.error(f"Gagal mendapatkan prediksi: {result.get('detail', 'Unknown error')}")

        except Exception as e:
            st.error(f"❌ Tidak dapat terhubung ke server API: {str(e)}")

with tab_batch:
    st.header("📂 Analisis Risiko Kredit Batch (CSV)")
    st.markdown("Unggah file CSV dengan data nasabah untuk mendapatkan analisis risiko secara massal.")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Tampilkan pratinjau data mentah yang diunggah
        st.write("**Pratinjau Data (5 Baris Pertama):**")
        df_preview = pd.read_csv(uploaded_file)
        st.dataframe(df_preview.head(), use_container_width=True)
        
        # Tombol untuk memicu proses di Backend
        if st.button("🔍 Mulai Analisis Batch", key="btn_batch"):
            try:
                # Reset pointer file ke awal agar bisa dibaca lagi untuk pengiriman
                uploaded_file.seek(0)
                
                # Siapkan file untuk dikirim via requests (multipart/form-data)
                files = {'file': (uploaded_file.name, uploaded_file.read(), 'text/csv')}
                
                with st.spinner(f'Sedang menganalisis {len(df_preview)} data nasabah... Mohon tunggu.'):
                    # Panggil Endpoint BATCH di FastAPI
                    response = requests.post("http://localhost:8000/predict-batch", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Analisis Selesai! Berhasil memproses {result['count']} data.")
                    
                    # Konversi data JSON kembali ke DataFrame
                    df_result = pd.DataFrame(result['data'])
                    
                    # Tampilkan pratinjau HASIL
                    st.subheader("📋 Pratinjau Hasil Prediksi")
                    # Tampilkan kolom ID (jika ada) dan kolom prediksi di depan
                    cols = df_result.columns.tolist()
                    if 'SK_ID_CURR' in cols: # Pindahkan ID ke depan
                        cols.insert(0, cols.pop(cols.index('SK_ID_CURR')))
                    
                    # Taruh PREDIKSI di urutan paling depan
                    cols.insert(1, cols.pop(cols.index('PRED_PROBABILITY')))
                    cols.insert(2, cols.pop(cols.index('PRED_RISK_RATING')))
                    
                    st.dataframe(df_result[cols].head(10), use_container_width=True)
                    
                    # 2. NEW: Tombol Unduh (Download Button)
                    st.divider()
                    st.subheader("💾 Unduh Hasil Lengkap")
                    
                    # Konversi DataFrame hasil ke CSV string
                    csv_string = df_result.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Unduh File CSV Hasil Prediksi",
                        data=csv_string,
                        file_name=f"hasil_prediksi_home_credit_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                else:
                    error_detail = response.json().get('detail', 'Terjadi kesalahan sistem.')
                    st.error(f"❌ Gagal memproses: {error_detail}")
                    
            except Exception as e:
                st.error(f"❌ Tidak dapat terhubung ke server API: {str(e)}")

# 5. Footer
st.divider()
st.caption("Credit Scoring Project - Developed by Baltasar (Informatics Engineer)")