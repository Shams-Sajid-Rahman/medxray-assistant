FROM python:3.13-slim

# opencv (used by grad-cam) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# CPU-only PyTorch first (the default build includes ~2 GB of GPU libraries)
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Chroma's embedding model now, so the app starts fast
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as D; D()(['warm up'])"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
