FROM python:3.10-slim

WORKDIR /app

# Sistem gereksinimleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Proje dosyalarını kopyala
COPY . .

# Hugging Face Spaces varsayılan portu 7860'tır
EXPOSE 7860

# Gunicorn ile 7860 portunda başlat
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
