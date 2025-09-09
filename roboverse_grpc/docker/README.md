# RoboVerse gRPC Docker Deployment

This directory contains Docker configuration files for deploying the RoboVerse gRPC infrastructure in containers.

## Architecture

The deployment consists of:

1. **Policy Server Container**: GPU-accelerated policy inference server
2. **RoboVerse Environment Containers**: Multiple environment instances connecting to the policy server
3. **Optional Services**: Monitoring, debugging, and logging components

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU support)

### Basic Deployment

1. **Build and start services:**
   ```bash
   cd roboverse_grpc/docker
   docker-compose up --build
   ```

2. **Run with multiple environments:**
   ```bash
   docker-compose --profile multi-env up --build
   ```

3. **Enable debugging tools:**
   ```bash
   docker-compose --profile debug up --build
   ```

### Environment Variables

Key environment variables can be set in `.env` or passed directly:

```bash
# Set policy type and model
export POLICY_TYPE=diffusion
export MODEL_PATH=/workspace/models/my_policy.pth

# Set task configuration
export TASK_NAME=square_d0
export NUM_ENVS=8
export SIM=isaaclab

# Start with custom configuration
docker-compose up
```

## Configuration Files

### `.env`
Default environment variables for all services.

### `docker-compose.yml`
Main service orchestration file with:
- **policy-server**: GPU-enabled policy inference
- **roboverse-env-***: Environment containers
- **grpc-ui**: Web UI for gRPC debugging (port 8080)
- **fluentd**: Log aggregation

### Dockerfiles

- **`policy.Dockerfile`**: Policy server with ML/AI dependencies
- **`roboverse.Dockerfile`**: RoboVerse environment with gRPC client

## Usage Examples

### Single Environment with Diffusion Policy

```bash
export POLICY_TYPE=diffusion
export MODEL_PATH=/path/to/diffusion_model.pth
export TASK_NAME=square_d0
docker-compose up policy-server roboverse-env-1
```

### Multiple Environments with VLA Policy

```bash
export POLICY_TYPE=vla
export MODEL_PATH=/path/to/vla_model.pth
export TASK_NAME=stack_d0
docker-compose --profile multi-env up
```

### Development with Debug Tools

```bash
docker-compose --profile debug up
# Access gRPC UI at http://localhost:8080
```

## Volume Mounts

The containers mount several directories:

- **`../models`**: Pre-trained model files (read-only for policy server)
- **`../data`**: Dataset and configuration files
- **`../logs`**: Log files and outputs
- **`/tmp/.X11-unix`**: X11 socket for GUI rendering (Linux)

Create these directories before starting:

```bash
mkdir -p ../models ../data ../logs
```

## GPU Configuration

The policy server requires GPU access. Ensure:

1. **NVIDIA Container Toolkit is installed:**
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Specify GPU device:**
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # Use first GPU
   docker-compose up
   ```

## Monitoring and Debugging

### Health Checks

Services include health checks to ensure proper startup order:

```bash
# Check service health
docker-compose ps

# View service logs
docker-compose logs policy-server
docker-compose logs roboverse-env-1
```

### gRPC UI

Enable the debug profile to access a web-based gRPC client:

```bash
docker-compose --profile debug up
# Navigate to http://localhost:8080
```

### Performance Monitoring

Monitor resource usage:

```bash
# Container stats
docker stats

# GPU usage
nvidia-smi

# Policy server metrics
docker-compose exec policy-server nvidia-smi
```

## Scaling

### Horizontal Scaling

Add more environment containers by extending the docker-compose file:

```yaml
roboverse-env-3:
  extends:
    service: roboverse-env-1
  container_name: roboverse_env_3
  environment:
    - ENV_ID=2
```

### Multiple Policy Servers

Run multiple policy servers with load balancing:

```yaml
policy-server-2:
  extends:
    service: policy-server
  container_name: roboverse_policy_server_2
  ports:
    - "50052:50051"
  environment:
    - GRPC_SERVER_PORT=50051
    - CUDA_VISIBLE_DEVICES=1
```

## Troubleshooting

### Common Issues

1. **GPU not detected:**
   - Verify NVIDIA Container Toolkit installation
   - Check `docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi`

2. **Connection refused:**
   - Ensure policy server is healthy before starting environments
   - Check firewall settings for port 50051

3. **Out of memory:**
   - Reduce `BATCH_SIZE` in policy server
   - Reduce `NUM_ENVS` per environment container

4. **Display/rendering issues:**
   - Set correct `DISPLAY` environment variable
   - Ensure X11 forwarding permissions: `xhost +local:`

### Debug Commands

```bash
# Interactive shell in policy server
docker-compose exec policy-server bash

# View policy server logs
docker-compose logs -f policy-server

# Test gRPC connection
docker-compose exec roboverse-env-1 python -c "
from roboverse_grpc.client.robosuite_policy_client import *
config = RobosuitePolicyClientConfig(server_uri='policy-server:50051')
client = config.create()
print('Connection successful:', client.get_policy_metadata())
"
```

## Production Deployment

For production deployments:

1. **Use specific image tags** instead of building from source
2. **Configure persistent volumes** for models and data
3. **Set up proper logging** with log rotation
4. **Configure monitoring** with Prometheus/Grafana
5. **Use Docker Swarm or Kubernetes** for orchestration
6. **Set resource limits** for each service

Example production override:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  policy-server:
    image: roboverse/policy-server:v1.0.0
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```