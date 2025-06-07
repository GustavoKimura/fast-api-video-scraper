FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV TORCH_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libgtk-3-0 libxss1 libasound2 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libegl1 \
    ca-certificates wget curl gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install playwright && playwright install chromium

COPY . .

RUN mkdir -p /models && \
    python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
