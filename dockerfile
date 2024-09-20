# Use an official PyTorch image as the base
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set up the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    tesseract-ocr \
    poppler-utils \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Clone pix2tex from GitHub
RUN git clone --depth 1 https://github.com/lukas-blecher/LaTeX-OCR.git pix2tex

WORKDIR /app/pix2tex

# Install pix2tex dependencies
RUN pip install --no-cache-dir \
    numpy \
    pillow \
    opencv-python \
    tqdm \
    transformers==4.11.3 \
    timm==0.4.12 \
    torch \
    torchvision \
    python-docx==0.8.11 \
    watchdog \
    albumentations \
    editdistance \
    rapidfuzz \
    pyspellchecker \
    pix2tex

# Copy your Python script into the container
COPY monitor_and_transcribe.py /app/

# Create necessary directories
RUN mkdir -p /mnt/d/OCR-stuff/midterm-prep /mnt/d/OCR-stuff/midterm-prep-output

# Download the pix2tex model
RUN python3 -c "from pix2tex.cli import LatexOCR; LatexOCR()"

# Set up the entrypoint to run your monitoring script
CMD ["python3", "/app/monitor_and_transcribe.py"]