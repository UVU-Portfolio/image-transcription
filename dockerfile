# Use an official PyTorch image as the base
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set up the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Clone pix2tex from GitHub at the specific commit
RUN git clone https://github.com/lukas-blecher/LaTeX-OCR.git pix2tex && \
    cd pix2tex && \
    git checkout 3bd6f9a

WORKDIR /app/pix2tex

# Copy application files
COPY requirements.txt monitor_and_transcribe.py /app/

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir typing_extensions>=4.0.0 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install -e .

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Set the model download script
COPY download_model.py /app/
RUN python3 /app/download_model.py

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set up healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import os; exit(0 if os.path.exists('/app/pix2tex/run.py') else 1)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Set up the entrypoint
ENTRYPOINT ["python3", "/app/monitor_and_transcribe.py"]