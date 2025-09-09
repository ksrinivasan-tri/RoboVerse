# Dockerfile for Policy Server with GPU support
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

# Create conda environment for policy inference
RUN conda create -n policy python=${PYTHON_VERSION} -y
RUN echo "source activate policy" >> ~/.bashrc

# Activate environment for subsequent RUN commands
SHELL ["conda", "run", "-n", "policy", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install ML/AI packages for different policy types
RUN pip install \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    pillow \
    tqdm \
    loguru \
    transformers \
    diffusers \
    accelerate \
    grpcio \
    grpcio-tools \
    protobuf \
    einops \
    gymnasium

# Install additional packages for specific policy types
RUN pip install \
    stable-baselines3 \
    rsl-rl \
    wandb \
    tensorboard

# Set working directory
WORKDIR /workspace

# Copy RoboVerse gRPC components
COPY ./roboverse_grpc /workspace/roboverse_grpc
COPY ./metasim /workspace/metasim

# Install gRPC components
RUN pip install -e ./roboverse_grpc

# Create directories for models, data, and logs
RUN mkdir -p /workspace/models /workspace/data /workspace/logs

# Set default environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV GRPC_SERVER_PORT=50051
ENV GRPC_SERVER_HOST=0.0.0.0

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import grpc; from roboverse_grpc.proto import RobosuitePolicy_pb2_grpc; \
         channel = grpc.insecure_channel('localhost:50051'); \
         stub = RobosuitePolicy_pb2_grpc.HealthStub(channel); \
         stub.Check(RobosuitePolicy_pb2.HealthCheckRequest())" || exit 1

# Default command to run policy server
CMD ["python", "-m", "roboverse_grpc.server.robosuite_policy_server"]

# Set entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "policy"]
