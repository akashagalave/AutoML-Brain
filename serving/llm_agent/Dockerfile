FROM python:3.11-slim

WORKDIR /app

COPY serving/llm_agent ./llm_agent
COPY reports ./reports

RUN pip install --no-cache-dir openai requests reportlab

WORKDIR /app/llm_agent

CMD ["python", "reporting_agent.py"]