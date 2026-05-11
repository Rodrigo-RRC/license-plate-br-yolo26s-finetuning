FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    ultralytics>=8.4.36 \
    torch>=2.11.0 \
    torchvision>=0.26.0 \
    torchmetrics>=1.9.0

COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "src/eval_por_tipo_best_placas.py"]
