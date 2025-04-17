FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install torch and related packages first (cached if possible)
RUN pip install --no-cache-dir \
    torch==2.6.0+cu126 \
    torchvision==0.21.0+cu126 \
    torchaudio==2.6.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the backend code (including updated app.py)
COPY . .

# Ensure uploads and model folders exist
RUN mkdir -p uploads model

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/model \
    HF_HOME=/app/model

EXPOSE 5000

CMD ["python", "app.py"]
