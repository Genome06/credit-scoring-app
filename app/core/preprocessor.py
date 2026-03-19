import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Union

class DataTransformer:
    def __init__(self, scaler_path: str):
        """
        Initialize OOP-based data transformation object.
        """
        print(f"📂 Loading Scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)
        
        # Order of 26 mandatory features (adjusted to your info() list)
        self.final_features_list = [
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCES_MEAN', 
            'EXT_SOURCES_NAN_COUNT', 'AGE', 'DAYS_EMPLOYED_ANOM', 'REGION_RATING_CLIENT_W_CITY',
            'NAME_EDUCATION_TYPE_Academic degree', 'NAME_EDUCATION_TYPE_Higher education',
            'NAME_EDUCATION_TYPE_Incomplete higher', 'NAME_EDUCATION_TYPE_Lower secondary',
            'NAME_EDUCATION_TYPE_Secondary / secondary special',
            'NAME_INCOME_TYPE_Commercial associate', 'NAME_INCOME_TYPE_Other',
            'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Working',
            'INCOME_CREDIT_PERC_LOG', 'AMT_ANNUITY_LOG', 'GOODS_RATIO_LOG', 
            'ANNUITY_INCOME_PERC_LOG', 'AMT_CREDIT_LOG', 'PAYMENT_RATE_LOG',
            'DAYS_EMPLOYED_PERC_LOG', 'DAYS_EMPLOYED_ABS'
        ]

        # 13 Features that are indeed fitted to Scaler
        self.num_cols_to_scale = [
            'INCOME_CREDIT_PERC_LOG', 'DAYS_EMPLOYED_PERC_LOG', 'AMT_ANNUITY_LOG', 
            'GOODS_RATIO_LOG', 'ANNUITY_INCOME_PERC_LOG', 'AMT_CREDIT_LOG', 
            'PAYMENT_RATE_LOG', 'AGE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 
            'EXT_SOURCE_3', 'EXT_SOURCES_MEAN', 'DAYS_EMPLOYED_ABS'
        ]
        
        # Median from training data (According to what you entered earlier)
        self.training_medians = {
            'EXT_SOURCE_1': 0.5, 'EXT_SOURCE_2': 0.56, 'EXT_SOURCE_3': 0.53,
            'AMT_ANNUITY': 24903.0, 'AMT_CREDIT': 513531.0, 'DAYS_BIRTH': -15750,
            'DAYS_EMPLOYED': -1648, 'REGION_RATING_CLIENT_W_CITY': 2, 
            'AMT_INCOME_TOTAL': 147150.0, 'AMT_GOODS_PRICE': 450000.0
        }
        
        self.cols_to_log = [
            'INCOME_CREDIT_PERC', 'AMT_ANNUITY', 'GOODS_RATIO', 
            'ANNUITY_INCOME_PERC', 'AMT_CREDIT', 'PAYMENT_RATE', 'DAYS_EMPLOYED_PERC' 
        ]
        self.valid_incomes = ['Working', 'Commercial associate', 'Pensioner', 'State servant']

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper method for basic numeric feature engineering & ratios."""
        df = df.copy()
        
        # Basic Features
        df['EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['EXT_SOURCES_NAN_COUNT'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].isnull().sum(axis=1)
        df['AGE'] = df['DAYS_BIRTH'] / -365
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        df['DAYS_EMPLOYED_ABS'] = df['DAYS_EMPLOYED'].abs()
        
        # Financial Ratios (Preventing division by zero)
        epsilon = 1e-6 # Small number so no error div by zero
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + epsilon)
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + epsilon)
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + epsilon)
        df['GOODS_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + epsilon)
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + epsilon)
        
        return df

    def _log_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper method for log transformation (log1p)."""
        df = df.copy()
        for col in self.cols_to_log:
            # Using max(0, val) so log1p doesn't error for unexpected negative input
            df[f'{col}_LOG'] = np.log1p(df[col].apply(lambda x: max(0, x)))
        return df

    def _categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper method for grouping and OHE."""
        df = df.copy()
        
        # Grouping Income Type
        df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].apply(
            lambda x: x if x in self.valid_incomes else 'Other'
        )
        
        # One-Hot Encoding
        df = pd.get_dummies(df, columns=['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE'], dtype=int)
        return df

    def transform(self, raw_data_list: List[Dict]) -> np.ndarray:
        """
        Main method to process raw data list from API into scaled numpy array.
        """
        print(f"🛠️ Processing {len(raw_data_list)} customer data...")
        
        df = pd.DataFrame(raw_data_list)
        
        # 1. Median Imputation (Raw Features)
        for col, median in self.training_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)
        
        # 2. Transformation Pipeline
        df = self._feature_engineering(df)
        df = self._categorical_encoding(df)
        df = self._log_transformation(df)
        
        # 3. 26 Columns Alignment
        X_final = df.reindex(columns=self.final_features_list)
        
        # 4. Safety Net: Only fill 0 for columns that are STILL NaN (usually dummy columns)
        # This won't damage the median because median has already entered in stage 1
        X_final = X_final.fillna(0)
        
        # 5. Partial Scaling (KEY FIX FOR 13 vs 26 ERROR)
        # We only scale 13 columns, but the results we still save in the 26-column dataframe
        X_final[self.num_cols_to_scale] = self.scaler.transform(X_final[self.num_cols_to_scale])
        
        print(f"✅ Data ready. Shape: {X_final.shape}") # Must be (n, 26)
        return X_final.values