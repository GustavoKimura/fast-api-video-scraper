FROM mcr.microsoft.com/playwright/python:v1.43.1-jammy

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    playwright install --with-deps

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
