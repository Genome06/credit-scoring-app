import joblib
import numpy as np
import pandas as pd
from app.core.preprocessor import DataTransformer

class CreditPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """
        Inisialisasi predictor dengan memuat model dan preprocessor.
        """
        print(f"🚀 Memuat Model dari {model_path}...")
        self.model = joblib.load(model_path)
        
        # Inisialisasi DataTransformer (OOP Component sebelumnya)
        self.transformer = DataTransformer(scaler_path)

    def _get_analysis_result(self, prob: float) -> dict:
        """
        Logic pusat untuk menentukan Rating dan Pesan Informatif.
        """
        if prob < 0.3:
            return {
                "rating": "Low Risk (Lancar)",
                "message": (
                    "✅ **Profil Sangat Baik.** Nasabah menunjukkan indikator stabilitas finansial yang kuat. "
                    "Skor kredit eksternal berada di atas rata-rata dan beban cicilan (Payment Rate) "
                    "tergolong rendah. Pengajuan ini sangat direkomendasikan untuk disetujui (Auto-Approval)."
                )
            }
        elif prob < 0.6:
            return {
                "rating": "Medium Risk (Waspada)",
                "message": (
                    "⚠️ **Tinjauan Manual Diperlukan.** Profil nasabah berada di ambang batas aman. "
                    "Terdapat sedikit fluktuasi pada skor eksternal atau rasio hutang yang mulai meningkat. "
                    "Disarankan untuk melakukan verifikasi dokumen pendapatan tambahan atau mempertimbangkan "
                    "tenor pinjaman yang lebih pendek."
                )
            }
        else:
            return {
                "rating": "High Risk (Macet)",
                "message": (
                    "🚨 **Peringatan Risiko Tinggi.** Berdasarkan analisis pola data, nasabah memiliki probabilitas "
                    "gagal bayar yang signifikan. Hal ini biasanya dipicu oleh beban cicilan yang terlalu berat "
                    "dibandingkan pendapatan atau rekam jejak kredit eksternal yang lemah. "
                    "Sangat disarankan untuk melakukan penolakan atau meminta jaminan tambahan."
                )
            }
    
    # --- NEW: Helper untuk Batch (Vectorized) ---
    def _get_risk_rating_batch(self, prob: float) -> str:
        """Helper simpel untuk digunakan dengan .apply() pada series."""
        if prob < 0.3: return "Low Risk"
        elif prob < 0.6: return "Medium Risk"
        else: return "High Risk"

    # --- NEW: Method untuk prediksi BATCH (CSV/DataFrame) ---
    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Menerima DataFrame mentah, memprosesnya, dan mengembalikan 
        DataFrame yang sudah berisi hasil prediksi (Probability & Rating).
        """
        print(f"📊 Memproses Batch Prediction untuk {len(df_input)} baris...")
        
        # 1. Konversi DF ke List[Dict] agar bisa diproses preprocessor OOP kita
        raw_data_list = df_input.to_dict('records')
        
        # 2. Transformasi Data (26 kolom + Scaling) - Ini sudah vectorized di dalam class
        X_scaled = self.transformer.transform(raw_data_list)
        
        # 3. Prediksi Probabilitas (Vectorized)
        # Menggunakan numpy array slicing [:, 1] sangat cepat
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # 4. Gabungkan hasil ke DataFrame output
        df_output = df_input.copy()
        df_output['PRED_PROBABILITY'] = np.round(probs, 4)
        
        # 5. Tentukan Rating (Vectorized apply)
        # Catatan: Kita tidak memberikan 'message' detail di CSV agar file tidak terlalu besar
        df_output['PRED_RISK_RATING'] = df_output['PRED_PROBABILITY'].apply(self._get_risk_rating_batch)
        
        print("✅ Batch Prediction selesai.")
        return df_output

    def predict(self, input_data: dict) -> dict:
        """
        Method utama yang memanggil logic pesan detail.
        """
        X_scaled = self.transformer.transform([input_data])
        prob = self.model.predict_proba(X_scaled)[0, 1]
        
        # Ambil rating dan pesan berdasarkan probabilitas
        analysis = self._get_analysis_result(prob)
        
        return {
            "probability": round(float(prob), 4),
            "risk_rating": analysis["rating"],
            "message": analysis["message"]
        }