# Use lightweight Python image
FROM python:3.10-slim

# Hugging Face default port
ENV PORT=7860
EXPOSE 7860

WORKDIR /code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Create non-root user for security (mandatory on Hugging Face)
RUN useradd -m appuser && chown -R appuser /code
USER appuser

# Helper script to run FastAPI & Streamlit simultaneously
# We need to create this run.sh file in the root project
CMD ["sh", "run.sh"]