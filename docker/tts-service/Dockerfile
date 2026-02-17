# Dockerfile for Coqui TTS Service (Python 3.10)
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Coqui TTS
RUN pip install --no-cache-dir TTS flask

# Copy service script
COPY tts_service.py /app/

# Expose port
EXPOSE 5000

# Run the service
CMD ["python", "tts_service.py"]
