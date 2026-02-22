FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install dvc[s3] mlflow dagshub

COPY . .

CMD ["bash", "-c", "dvc pull && dvc repro && dvc push"]