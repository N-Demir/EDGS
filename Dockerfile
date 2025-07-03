# Use PyTorch CUDA image as base
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Makes debugging easier because python logs are sent immediately
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# First part is necessary for the gcloud cli to get installed
RUN apt-get update && apt-get install -y curl gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y \
    google-cloud-cli \
    git \
    wget \
    unzip \
    cmake \
    build-essential \
    ninja-build \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libglm-dev \
    tmux \
    && rm -rf /var/lib/apt/lists/*
# final rm here reduces Docker image size

WORKDIR /root