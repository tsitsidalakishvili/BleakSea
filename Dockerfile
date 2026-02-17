# Use a Debian-based slim image with Python 3.10
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies, including OpenGL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libopenblas-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements.txt first to leverage Docker's caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8080

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
