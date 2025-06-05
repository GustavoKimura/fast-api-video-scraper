FROM python:3.11-slim

RUN apt-get update && apt-get install -y wget gnupg curl fonts-liberation libatk-bridge2.0-0 libgtk-3-0 libxss1 libasound2 libnss3 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 libx11-xcb1 libx11-6 libxext6 libxcb1 libxi6 libgl1 libegl1 ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install playwright && playwright install chromium

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
