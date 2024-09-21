FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    libcudnn8

# Set work directory
WORKDIR /code

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Install WhisperX
RUN pip3 install git+https://github.com/m-bain/whisperx.git

# Install Unidic
RUN python3 -m unidic download

# Copy project files
COPY ./main.py .

# Command to run the application
CMD ["fastapi", "run", "main.py", "--port", "80"]
