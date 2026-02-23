FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY serving/deployment_agent ./deployment_agent
COPY serving/dev-requirements.txt .

RUN pip install --no-cache-dir -r dev-requirements.txt \
    && pip install kubernetes mlflow dagshub

CMD ["python", "-m", "deployment_agent.deployment_agent"]