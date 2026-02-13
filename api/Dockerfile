FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential libaio-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api /app/api
COPY data /app/data

WORKDIR /app
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
