#!/bin/bash

# Jalankan Backend FastAPI di background (Port 8000 internal)
echo "🚀 Memulai FastAPI Backend..."
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

# Tunggu sebentar agar FastAPI siap
sleep 5

# Jalankan Streamlit Frontend di foreground (Port 7860 external)
echo "🎨 Memulai Streamlit Frontend di Port $PORT..."
streamlit run app/frontend/ui.py --server.port $PORT --server.address 0.0.0.0