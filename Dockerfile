# Use a specific Python version
FROM python:3.11.10-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application code into the container
COPY . .

# Build Cython extensions
RUN cd utils && python setup.py build_ext --inplace

# Create necessary directories if they don't exist
RUN mkdir -p data/models

# Expose the port for FastAPI application
EXPOSE 8000

# Specify the command to run the application with Gunicorn
CMD ["gunicorn", "app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]