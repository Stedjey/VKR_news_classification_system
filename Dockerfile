FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_DB_DIR=/app/db \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app ./app
COPY rubert_final_model ./rubert_final_model
COPY rubert_label_encoder.pkl ./rubert_label_encoder.pkl

RUN mkdir -p /app/db /app/.cache/huggingface

EXPOSE 8501

CMD ["sh", "-c", "python app/bot.py & streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501"]
