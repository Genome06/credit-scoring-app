#!/bin/bash

# Run FastAPI Backend in background (Port 8000 internal)
echo "🚀 Starting FastAPI Backend..."
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for FastAPI to be ready
sleep 5

# Run Streamlit Frontend in foreground (Port 7860 external)
echo "🎨 Starting Streamlit Frontend on Port $PORT..."
streamlit run app/frontend/ui.py --server.port $PORT --server.address 0.0.0.0