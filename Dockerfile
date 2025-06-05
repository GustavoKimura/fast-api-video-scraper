FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl gnupg \
    libnss3 libgtk-3-0 libxss1 libasound2 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libegl1 \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install playwright && playwright install chromium

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
