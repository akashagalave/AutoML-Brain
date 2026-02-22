FROM python:3.11-slim

WORKDIR /app

COPY dev1-requirements.txt .
RUN pip install --no-cache-dir -r dev1-requirements.txt

COPY . .

CMD ["python", "main.py"]