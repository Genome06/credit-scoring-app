# Gunakan image Python yang ringan
FROM python:3.10-slim

# Hugging Face default port
ENV PORT=7860
EXPOSE 7860

WORKDIR /code

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh app
COPY . .

# Buat user non-root untuk keamanan (wajib di Hugging Face)
RUN useradd -m appuser && chown -R appuser /code
USER appuser

# Skrip pembantu untuk menjalankan FastAPI & Streamlit sekaligus
# Kita perlu membuat file run.sh ini di root project
CMD ["sh", "run.sh"]