# Dockerfile for RoboVerse environment with gRPC client
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=11.8
ENV PYTHON_VERSION=3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create conda environment
RUN conda create -n roboverse python=${PYTHON_VERSION} -y
RUN echo "source activate roboverse" >> ~/.bashrc

# Activate environment for subsequent RUN commands
SHELL ["conda", "run", "-n", "roboverse", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install common Python packages
RUN pip install \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    pillow \
    tqdm \
    loguru \
    tyro \
    gymnasium \
    grpcio \
    grpcio-tools \
    protobuf

# Set working directory
WORKDIR /workspace

# Copy RoboVerse source code
COPY . /workspace/

# Install RoboVerse and dependencies
RUN pip install -e .

# Install gRPC dependencies
RUN pip install -e ./roboverse_grpc

# Create directories for data and logs
RUN mkdir -p /workspace/data /workspace/logs

# Set default command
CMD ["python", "-m", "roboverse_grpc.examples.run_client_example"]

# Expose default gRPC port
EXPOSE 50051

# Set entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "roboverse"]
