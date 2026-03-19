import streamlit as st
import requests
import pandas as pd
import io

# 1. Page Configuration
st.set_page_config(
    page_title="Home Credit Score Card",
    page_icon="💳",
    layout="wide"
)

# 2. Title & Header
st.title("💳 Home Credit Risk Analysis")
st.markdown("Choose an analysis method: Individual Input or Upload Batch (CSV).")

# --- 3. NEW: TABS SYSTEM ---
tab_single, tab_batch = st.tabs(["👤 Individual Analysis", "📂 Batch Analysis (CSV)"])

with tab_single:
    st.header("👤 Individual Credit Risk Analysis")
    st.markdown("Enter customer data to get real-time credit risk analysis.") 
    # 3. Customer Data Input Form
    # We divide into 2 columns so it's not too long downwards
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 External Scores & Rating")
        ext_1 = st.slider("EXT_SOURCE_1 (Credit Score 1)", 0.0, 1.0, 0.5)
        ext_2 = st.slider("EXT_SOURCE_2 (Credit Score 2)", 0.0, 1.0, 0.5)
        ext_3 = st.slider("EXT_SOURCE_3 (Credit Score 3)", 0.0, 1.0, 0.5)
        region_rating = st.selectbox("Region Rating (City)", [1, 2, 3], index=1)

        st.subheader("🎓 Customer Profile")
        education = st.selectbox("Education Level", [
            "Higher education", 
            "Secondary / secondary special", 
            "Incomplete higher", 
            "Lower secondary", 
            "Academic degree"
        ])
        income_type = st.selectbox("Income Type", [
            "Working", 
            "Commercial associate", 
            "Pensioner", 
            "State servant", 
            "Student", 
            "Unemployed"
        ])

    with col2:
        st.subheader("💰 Financial Data")
        amt_credit = st.number_input("Total Loan Amount (AMT_CREDIT)", min_value=0.0, value=500000.0)
        amt_annuity = st.number_input("Annual Annuity (AMT_ANNUITY)", min_value=0.0, value=25000.0)
        amt_income = st.number_input("Annual Income (AMT_INCOME)", min_value=0.0, value=150000.0)
        amt_goods = st.number_input("Goods Price (AMT_GOODS_PRICE)", min_value=0.0, value=450000.0)

        st.subheader("⏳ History & Age")
        age_years = st.number_input("Age (Years)", min_value=18, max_value=100, value=30)
        days_birth = age_years * -365 # Convert to dataset format

        emp_years = st.number_input("Years Employed (Years)", min_value=0, max_value=60, value=5)
        days_employed = emp_years * -365 # Convert to dataset format

    # 4. Prediction Button Logic
    if st.button("🔍 Analyze Credit Risk", use_container_width=True):
        # Prepare payload according to Pydantic Schema in FastAPI
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
            # Call FastAPI (localhost because one Docker container)
            with st.spinner('Calculating risk score...'):
                response = requests.post("http://localhost:8000/predict", json=payload)
                result = response.json()

            if response.status_code == 200:
                st.success("✅ Analysis Complete!")
                
                # Display Results
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Probability of Default", f"{result['probability'] * 100:.2f} %")
                    
                    # Alert Color based on Rating
                    rating = result['risk_rating']
                    if "Low" in rating:
                        st.info(f"Category: {rating}")
                    elif "Medium" in rating:
                        st.warning(f"Category: {rating}")
                    else:
                        st.error(f"Category: {rating}")
                
                with res_col2:
                    st.write("**System Analysis:**")
                    # Use markdown so bold and emoji appear correctly
                    st.markdown(result['message'])
                    
                # Progress Bar as "Risk Meter"
                st.progress(result['probability'])
                
            else:
                st.error(f"Failed to get prediction: {result.get('detail', 'Unknown error')}")

        except Exception as e:
            st.error(f"❌ Cannot connect to API server: {str(e)}")

with tab_batch:
    st.header("📂 Batch Credit Risk Analysis (CSV)")
    st.markdown("Upload CSV file with customer data to get mass risk analysis.")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Display raw data preview uploaded
        st.write("**Data Preview (First 5 Rows):**")
        df_preview = pd.read_csv(uploaded_file)
        st.dataframe(df_preview.head(), use_container_width=True)
        
        # Button to trigger process in Backend
        if st.button("🔍 Start Batch Analysis", key="btn_batch"):
            try:
                # Reset file pointer to beginning so it can be read again for sending
                uploaded_file.seek(0)
                
                # Prepare file for sending via requests (multipart/form-data)
                files = {'file': (uploaded_file.name, uploaded_file.read(), 'text/csv')}
                
                with st.spinner(f'Analyzing {len(df_preview)} customer data... Please wait.'):
                    # Call BATCH Endpoint in FastAPI
                    response = requests.post("http://localhost:8000/predict-batch", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Analysis Complete! Successfully processed {result['count']} data.")
                    
                    # Convert JSON data back to DataFrame
                    df_result = pd.DataFrame(result['data'])
                    
                    # Display RESULT preview
                    st.subheader("📋 Prediction Results Preview")
                    # Display ID column (if any) and prediction columns in front
                    cols = df_result.columns.tolist()
                    if 'SK_ID_CURR' in cols: # Move ID to front
                        cols.insert(0, cols.pop(cols.index('SK_ID_CURR')))
                    
                    # Put PREDICTION in the frontmost order
                    cols.insert(1, cols.pop(cols.index('PRED_PROBABILITY')))
                    cols.insert(2, cols.pop(cols.index('PRED_RISK_RATING')))
                    
                    st.dataframe(df_result[cols].head(10), use_container_width=True)
                    
                    # 2. NEW: Download Button
                    st.divider()
                    st.subheader("💾 Download Complete Results")
                    
                    # Convert result DataFrame to CSV string
                    csv_string = df_result.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Download Prediction Results CSV File",
                        data=csv_string,
                        file_name=f"home_credit_prediction_results_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                else:
                    error_detail = response.json().get('detail', 'System error occurred.')
                    st.error(f"❌ Failed to process: {error_detail}")
                    
            except Exception as e:
                st.error(f"❌ Cannot connect to API server: {str(e)}")

# 5. Footer
st.divider()
st.caption("Credit Scoring Project - Developed by Baltasar (Informatics Engineer)")