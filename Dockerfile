FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV APP_MODE=spaces

CMD ["sh", "-lc", "if [ \"$APP_MODE\" = \"api\" ]; then python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}; else python spaces_app.py; fi"]
