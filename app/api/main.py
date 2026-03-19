from fastapi import FastAPI, HTTPException, UploadFile, File # Add UploadFile & File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse # For custom response
from app.api.schemas import CreditInput, PredictionResponse
from app.core.predictor import CreditPredictor
import io
import os
import pandas as pd
import numpy as np

# 1. FastAPI Initialization
app = FastAPI(
    title="Home Credit Risk Scoring API",
    description="API to predict customer default probability using LightGBM Tuned.",
    version="1.0.0"
)

# 2. CORS Configuration (Cross-Origin Resource Sharing)
# Important so that Streamlit Frontend can communicate with FastAPI Backend without security barriers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, it's better to limit to certain domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Artifacts Path (Model & Scaler)
# We assume the folder structure matches what we created in the root project
MODEL_PATH = "artifacts/home_credit_lgbm_tuned.joblib"
SCALER_PATH = "artifacts/scaler_home_credit.joblib"

# Global Variable for Predictor (Singleton Pattern)
predictor = None

@app.on_event("startup")
async def startup_event():
    """
    Load model and scaler into memory when the application is first run.
    This saves latency time on each prediction request.
    """
    global predictor
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ ERROR: Model or scaler file not found in 'artifacts' folder!")
        return
    
    try:
        predictor = CreditPredictor(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
        print("✅ Model and Scaler successfully loaded. API ready to serve!")
    except Exception as e:
        print(f"❌ An error occurred while loading the model: {e}")

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
    Main endpoint to receive customer data and return risk score.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not ready or failed to load.")
    
    try:
        # Convert Pydantic Model to Python Dictionary
        # Pydantic v2 uses .model_dump(), v1 uses .dict()
        input_dict = data.model_dump() 
        
        # Execute Prediction through Predictor Layer
        prediction_results = predictor.predict(input_dict)
        
        return prediction_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error occurred: {str(e)}")

# NEW: Endpoint for BATCH (CSV Upload)
@app.post("/predict-batch")
async def predict_risk_batch(file: UploadFile = File(...)):
    """
    Endpoint to receive CSV upload, process it, and 
    return prediction results in JSON format (which will later be converted by FE to CSV).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not ready.")
    
    # File Type Validation
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be in CSV format.")
    
    try:
        # Read CSV file content into Pandas DataFrame
        contents = await file.read()
        # Using io.BytesIO to read data stream without saving to disk
        df_input = pd.read_csv(io.BytesIO(contents))
        
        # Ensure minimum columns exist (Example: EXT_SOURCE_1 is mandatory)
        # As an informatics engineer, validating raw columns is important
        required_cols = ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH']
        if not all(col in df_input.columns for col in required_cols):
             raise HTTPException(status_code=422, detail=f"CSV lacks mandatory columns: {required_cols}")

        # 1. Execute Batch Prediction
        df_result = predictor.predict_batch(df_input)
        
        # 2. KEY FIX: Replace NaN with None to be JSON-compliant
        # We use .replace({np.nan: None}) because None will become 'null' in JSON
        df_final = df_result.replace({np.nan: None})
        
        # 3. Convert to JSON records
        result_json = df_final.to_dict(orient='records')
        
        return JSONResponse(content={
            "status": "success",
            "count": len(df_final),
            "data": result_json
        })
        
    except Exception as e:
        # Add print to backend console for clearer debugging
        print(f"Detail Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing error occurred: {str(e)}")