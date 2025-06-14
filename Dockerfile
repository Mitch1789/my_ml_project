# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifact and inference code
COPY artifacts/model.pkl ./model.pkl
COPY inference.py ./inference.py

# Expose and run
EXPOSE 8080
CMD ["python", "inference.py"]
