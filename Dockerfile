FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt .
COPY ./main.py .
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y transformers && \
    pip install --no-cache-dir git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
