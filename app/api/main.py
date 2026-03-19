from fastapi import FastAPI, HTTPException, UploadFile, File # Tambah UploadFile & File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse # Untuk custom response
from app.api.schemas import CreditInput, PredictionResponse
from app.core.predictor import CreditPredictor
import io
import os
import pandas as pd
import numpy as np

# 1. Inisialisasi FastAPI
app = FastAPI(
    title="Home Credit Risk Scoring API",
    description="API untuk memprediksi probabilitas gagal bayar nasabah menggunakan LightGBM Tuned.",
    version="1.0.0"
)

# 2. Konfigurasi CORS (Cross-Origin Resource Sharing)
# Penting agar Frontend Streamlit bisa berkomunikasi dengan Backend FastAPI tanpa hambatan keamanan
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Di produksi, sebaiknya batasi ke domain tertentu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Path Artifacts (Model & Scaler)
# Kita asumsikan struktur folder sesuai yang kita buat di root project
MODEL_PATH = "artifacts/home_credit_lgbm_tuned.joblib"
SCALER_PATH = "artifacts/scaler_home_credit.joblib"

# Variabel Global untuk Predictor (Singleton Pattern)
predictor = None

@app.on_event("startup")
async def startup_event():
    """
    Memuat model dan scaler ke memori saat aplikasi pertama kali dijalankan.
    Ini menghemat waktu latensi pada setiap request prediksi.
    """
    global predictor
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ ERROR: File model atau scaler tidak ditemukan di folder 'artifacts'!")
        return
    
    try:
        predictor = CreditPredictor(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
        print("✅ Model dan Scaler berhasil dimuat. API siap melayani!")
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat memuat model: {e}")

# 4. Endpoints
@app.get("/")
def root():
    """Health Check Endpoint."""
    return {
        "message": "Welcome to Home Credit Risk API",
        "status": "Running",
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(data: CreditInput):
    """
    Endpoint utama untuk menerima data nasabah dan mengembalikan skor risiko.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model belum siap atau gagal dimuat.")
    
    try:
        # Konversi Pydantic Model ke Python Dictionary
        # Pydantic v2 menggunakan .model_dump(), v1 menggunakan .dict()
        input_dict = data.model_dump() 
        
        # Eksekusi Prediksi lewat Layer Predictor
        prediction_results = predictor.predict(input_dict)
        
        return prediction_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {str(e)}")

# NEW: Endpoint untuk BATCH (CSV Upload)
@app.post("/predict-batch")
async def predict_risk_batch(file: UploadFile = File(...)):
    """
    Endpoint untuk menerima unggahan CSV, memprosesnya, dan 
    mengembalikan hasil prediksi dalam format JSON (yang nanti diubah FE jadi CSV).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    
    # Validasi Tipe File
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File harus berformat CSV.")
    
    try:
        # Baca konten file CSV menjadi Pandas DataFrame
        contents = await file.read()
        # Menggunakan io.BytesIO untuk membaca stream data tanpa menyimpannya ke disk
        df_input = pd.read_csv(io.BytesIO(contents))
        
        # Pastikan kolom minimal ada (Contoh: EXT_SOURCE_1 wajib ada)
        # Sebagai informatics, validasi kolom mentah itu penting
        required_cols = ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH']
        if not all(col in df_input.columns for col in required_cols):
             raise HTTPException(status_code=422, detail=f"CSV kekurangan kolom wajib: {required_cols}")

        # 1. Eksekusi Prediksi Batch
        df_result = predictor.predict_batch(df_input)
        
        # 2. KUNCI PERBAIKAN: Ganti NaN menjadi None agar JSON-compliant
        # Kita menggunakan .replace({np.nan: None}) karena None akan berubah jadi 'null' di JSON
        df_final = df_result.replace({np.nan: None})
        
        # 3. Konversi ke JSON records
        result_json = df_final.to_dict(orient='records')
        
        return JSONResponse(content={
            "status": "success",
            "count": len(df_final),
            "data": result_json
        })
        
    except Exception as e:
        # Tambahkan print ke console backend untuk debugging yang lebih jelas
        print(f"Detail Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pemrosesan file: {str(e)}")