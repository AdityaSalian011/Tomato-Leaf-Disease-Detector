FROM python:3.11-slim

WORKDIR /app

# System deps needed by torchvision/Pillow for image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model weights
COPY main.py tomato_leaf_app.py mobilenet_tomato_leaf_detector.pt ./

# Render sets $PORT at runtime; default to 8000 for local testing
ENV PORT=8000
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}