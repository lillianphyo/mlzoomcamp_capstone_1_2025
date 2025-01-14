# Dockerfile
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y python3-distutils

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV DEBUG=False
ENV MODEL_PATH=artifacts/models/lstm_model.h5
ENV SCALER_PATH=artifacts/models/scaler.pkl
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "api/app.py"]