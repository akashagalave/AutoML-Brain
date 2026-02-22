# ==============================
# Base Image (Lightweight)
# ==============================
FROM python:3.11-slim

# ==============================
# Environment Variables
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# ==============================
# System Dependencies
# ==============================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# Working Directory
# ==============================
WORKDIR /app

# ==============================
# Copy Requirements First (Layer Caching)
# ==============================
COPY serving/dev-requirements.txt .

RUN pip install --no-cache-dir -r dev-requirements.txt

# ==============================
# Copy Application Code
# ==============================
COPY serving ./serving
COPY src ./src
COPY models ./models

# ==============================
# Expose Port
# ==============================
EXPOSE 8000

# ==============================
# Production Command
# ==============================
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]