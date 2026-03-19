import joblib
import numpy as np
import pandas as pd
from app.core.preprocessor import DataTransformer

class CreditPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize predictor by loading model and preprocessor.
        """
        print(f"🚀 Loading Model from {model_path}...")
        self.model = joblib.load(model_path)
        
        # Initialize DataTransformer (OOP Component previously)
        self.transformer = DataTransformer(scaler_path)

    def _get_analysis_result(self, prob: float) -> dict:
        """
        Central logic to determine Rating and Informative Message.
        """
        if prob < 0.3:
            return {
                "rating": "Low Risk (Smooth)",
                "message": (
                    "✅ **Very Good Profile.** Customer shows strong financial stability indicators. "
                    "External credit scores are above average and debt burden (Payment Rate) "
                    "is relatively low. This application is highly recommended for approval (Auto-Approval)."
                )
            }
        elif prob < 0.6:
            return {
                "rating": "Medium Risk (Caution)",
                "message": (
                    "⚠️ **Manual Review Required.** Customer profile is on the safe threshold. "
                    "There is slight fluctuation in external scores or debt ratio starting to increase. "
                    "It is recommended to verify additional income documents or consider shorter loan terms."
                )
            }
        else:
            return {
                "rating": "High Risk (Default)",
                "message": (
                    "🚨 **High Risk Warning.** Based on data pattern analysis, the customer has a significant probability of default. "
                    "This is usually triggered by debt burden that is too heavy compared to income or weak external credit history. "
                    "It is highly recommended to reject or request additional collateral."
                )
            }
    
    # --- NEW: Helper for Batch (Vectorized) ---
    def _get_risk_rating_batch(self, prob: float) -> str:
        """Simple helper to use with .apply() on series."""
        if prob < 0.3: return "Low Risk"
        elif prob < 0.6: return "Medium Risk"
        else: return "High Risk"

    # --- NEW: Method for BATCH prediction (CSV/DataFrame) ---
    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Receives raw DataFrame, processes it, and returns 
        DataFrame that already contains prediction results (Probability & Rating).
        """
        print(f"📊 Processing Batch Prediction for {len(df_input)} rows...")
        
        # 1. Convert DF to List[Dict] so it can be processed by our OOP preprocessor
        raw_data_list = df_input.to_dict('records')
        
        # 2. Data Transformation (26 columns + Scaling) - This is already vectorized inside the class
        X_scaled = self.transformer.transform(raw_data_list)
        
        # 3. Probability Prediction (Vectorized)
        # Using numpy array slicing [:, 1] is very fast
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # 4. Combine results to output DataFrame
        df_output = df_input.copy()
        df_output['PRED_PROBABILITY'] = np.round(probs, 4)
        
        # 5. Determine Rating (Vectorized apply)
        # Note: We do not provide detailed 'message' in CSV so the file is not too large
        df_output['PRED_RISK_RATING'] = df_output['PRED_PROBABILITY'].apply(self._get_risk_rating_batch)
        
        print("✅ Batch Prediction completed.")
        return df_output

    def predict(self, input_data: dict) -> dict:
        """
        Main method that calls detailed message logic.
        """
        X_scaled = self.transformer.transform([input_data])
        prob = self.model.predict_proba(X_scaled)[0, 1]
        
        # Get rating and message based on probability
        analysis = self._get_analysis_result(prob)
        
        return {
            "probability": round(float(prob), 4),
            "risk_rating": analysis["rating"],
            "message": analysis["message"]
        }