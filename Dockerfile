FROM python:3.11-slim

# Install FFmpeg and CA certificates, then clean up apt cache to keep image minimal
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create Hugging Face Spaces compatible non-root user (UID 1000)
RUN useradd -m -u 1000 user

# Prepare application and writable temporary directories for UID 1000
RUN mkdir -p /home/user/app /tmp/downloads && \
    chown -R user:user /home/user/app /tmp/downloads && \
    chmod 775 /tmp/downloads && \
    chmod 1777 /tmp

USER user

ENV HOME=/home/user \
    PATH="/home/user/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /home/user/app

# Install Python dependencies first for better Docker layer caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=user:user app.py .

EXPOSE 7860

CMD ["python", "app.py"]
