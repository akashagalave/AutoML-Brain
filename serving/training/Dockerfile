FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install dvc[s3] mlflow dagshub kubernetes

COPY . .

# Run DVC pipeline + registry + trigger deployment job
CMD ["bash", "-c", "\
    dvc pull && \
    dvc repro && \
    dvc push && \
    python serving/training/trigger_deployment.py \
"]