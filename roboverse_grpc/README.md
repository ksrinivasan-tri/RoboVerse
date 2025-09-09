# RoboVerse gRPC Infrastructure

A distributed gRPC-based architecture for running RoboVerse robotics simulations with remote policy inference servers. This system allows you to run physics simulations in one process/container while serving trained policies (diffusion models, VLA models, neural networks) from separate GPU-accelerated inference servers.

## Architecture Overview

```
┌─────────────────────┐    gRPC     ┌─────────────────────┐
│  RoboVerse Envs     │◄──────────►│  Policy Server      │
│                     │             │                     │
│  • Isaac Lab        │             │  • Diffusion Models │
│  • Isaac Gym        │             │  • VLA Models       │
│  • MuJoCo           │             │  • IsaacLab Policies│
│  • Robosuite Tasks  │             │  • Custom Policies  │
└─────────────────────┘             └─────────────────────┘
```

**Key Benefits:**
- **Scalability**: Multiple environments can share policy servers
- **Resource Efficiency**: Separate GPU allocation for simulation vs inference
- **Flexibility**: Mix different simulators with different policy types
- **Fault Tolerance**: Policies can restart independently of environments
- **Development Workflow**: Develop and test policies separately from environments

## Quick Start

### 1. Installation

```bash
# Clone RoboVerse
git clone https://github.com/your-org/RoboVerse.git
cd RoboVerse

# Install RoboVerse with gRPC support
pip install -e .
pip install grpcio grpcio-tools

# Install gRPC components
pip install -e ./roboverse_grpc
```

### 2. Start Policy Server

```bash
# Start a random policy server for testing
python -m roboverse_grpc.examples.start_policy_server_example \
    --policy-type random \
    --host 0.0.0.0 \
    --port 50051
```

### 3. Run RoboVerse Task with gRPC

```bash
# Run square_d0 task with remote policy
python -m roboverse_grpc.examples.run_robosuite_task_with_grpc \
    --task square_d0 \
    --sim isaaclab \
    --num-envs 4 \
    --policy-server localhost:50051
```

## Policy Types Supported

### Diffusion Policies
```bash
python -m roboverse_grpc.examples.start_policy_server_example \
    --policy-type diffusion \
    --model-path /path/to/diffusion_model.pth \
    --device cuda
```

### Vision-Language-Action (VLA) Models
```bash
python -m roboverse_grpc.examples.start_policy_server_example \
    --policy-type vla \
    --model-path /path/to/vla_model.pth \
    --device cuda
```

### IsaacLab Trained Policies
```bash
python -m roboverse_grpc.examples.start_policy_server_example \
    --policy-type isaaclab \
    --model-path /path/to/isaaclab_policy.pth \
    --device cuda
```

## Docker Deployment

### Single Container Setup
```bash
cd roboverse_grpc/docker

# Start policy server and environment
docker-compose up --build
```

### Multi-Environment Setup
```bash
# Start with multiple environment containers
docker-compose --profile multi-env up --build
```

### With Debugging Tools
```bash
# Include gRPC UI for debugging
docker-compose --profile debug up --build
# Access gRPC UI at http://localhost:8080
```

## Protocol Buffer Interface

The system uses Protocol Buffers for efficient serialization:

### Key Message Types

**RobosuiteObservation**: Contains robot states, object states, and camera data
```protobuf
message RobosuiteObservation {
  repeated RobotState robots = 1;
  repeated ObjectState objects = 2;  
  repeated CameraData cameras = 3;
  Header header = 4;
  int32 env_id = 5;
}
```

**RobosuiteAction**: Robot actions with joint targets
```protobuf
message RobosuiteAction {
  repeated RobotAction robot_actions = 1;
  Header header = 2;
  int32 env_id = 3;
}
```

### gRPC Services

- **PolicyStep**: Get actions from observations
- **PolicyReset**: Reset policy state  
- **GetPolicyMetadata**: Query policy information
- **Health**: Server health checking

## API Usage

### Client-Side Integration

```python
from roboverse_grpc.client.robosuite_policy_client import (
    RobosuitePolicyClient, RobosuitePolicyClientConfig
)

# Configure and create client
config = RobosuitePolicyClientConfig(
    server_uri="localhost:50051",
    wait_for_server=True
)
policy_client = config.create()

# Use in environment loop
observation = env.get_observation()  # RoboVerse EnvState
action = policy_client.step(observation)
env.step([action])
```

### Server-Side Policy Wrapper

```python
from roboverse_grpc.server.policy_wrappers.diffusion_wrapper import DiffusionPolicyWrapper
from roboverse_grpc.server.robosuite_policy_server import run_server, ServerConfig

# Load your trained model
model = load_diffusion_model("path/to/model.pth")

# Wrap model for gRPC serving
wrapper = DiffusionPolicyWrapper(model, device="cuda")

# Configure and start server
config = ServerConfig(host="0.0.0.0", port=50051, batch_size=16)
await run_server(wrapper, config)
```

## Configuration

### Environment Variables

```bash
# Policy server configuration
export POLICY_TYPE=diffusion
export MODEL_PATH=/path/to/model.pth
export GRPC_SERVER_PORT=50051
export DEVICE=cuda
export BATCH_SIZE=16

# Environment configuration  
export TASK_NAME=square_d0
export NUM_ENVS=4
export SIM=isaaclab
export POLICY_SERVER_URI=localhost:50051
```

### Server Configuration

```python
ServerConfig(
    host="0.0.0.0",
    port=50051,
    max_workers=10,
    batch_size=32,              # Batch size for efficient GPU usage
    batch_timeout_ms=50,        # Max latency vs throughput tradeoff
    enable_reflection=True,     # Enable gRPC reflection for debugging
    enable_health_check=True    # Enable health checking
)
```

## Performance Optimization

### Batching
The server automatically batches requests from multiple environments to improve GPU utilization:

```python
# Configure batching behavior
config = ServerConfig(
    batch_size=32,           # Process up to 32 observations together
    batch_timeout_ms=50      # Wait max 50ms to accumulate batch
)
```

### Resource Management

```python
# GPU memory management
policy_wrapper = DiffusionPolicyWrapper(
    model, 
    device="cuda:0"         # Specify GPU device
)

# Docker resource limits
services:
  policy-server:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Testing

### Unit Tests
```bash
# Run conversion tests
python -m pytest roboverse_grpc/tests/test_conversions.py -v

# Run all tests
python -m pytest roboverse_grpc/tests/ -v
```

### Integration Tests
```bash
# Test end-to-end pipeline
python -m roboverse_grpc.examples.start_policy_server_example --policy-type random &
sleep 5
python -m roboverse_grpc.examples.run_robosuite_task_with_grpc --episodes 1
```

## Monitoring and Debugging

### gRPC Reflection
Enable reflection for interactive debugging:
```bash
# Install grpcurl
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# List available services
grpcurl -plaintext localhost:50051 list

# Call methods interactively  
grpcurl -plaintext localhost:50051 roboverse_policy_interface.Health/Check
```

### Health Checking
```python
# Check server health
from roboverse_grpc.proto import RobosuitePolicy_pb2_grpc, RobosuitePolicy_pb2
import grpc

channel = grpc.insecure_channel('localhost:50051')
stub = RobosuitePolicy_pb2_grpc.HealthStub(channel)
response = stub.Check(RobosuitePolicy_pb2.HealthCheckRequest())
print(f"Status: {response.status}")
```

### Performance Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor container resources
docker stats

# View server logs
docker-compose logs -f policy-server
```

## Advanced Usage

### Custom Policy Wrappers

```python
from roboverse_grpc.server.policy_wrappers.base_wrapper import SingleInstancePolicyWrapper

class MyPolicyWrapper(SingleInstancePolicyWrapper):
    def __init__(self, my_model):
        super().__init__(my_model)
    
    def _step_single(self, observation):
        # Convert observation to your model's format
        model_input = self.convert_observation(observation)
        
        # Run inference
        model_output = self.policy(model_input)
        
        # Convert back to RoboVerse action format
        return self.convert_action(model_output, observation)
    
    def get_policy_metadata(self):
        return {
            "policy_type": "custom_policy",
            "model_name": "MyModel",
            "version": "1.0.0"
        }
```

### Multi-GPU Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  policy-server-gpu0:
    # ... config for GPU 0
    environment:
      - CUDA_VISIBLE_DEVICES=0
    ports:
      - "50051:50051"
  
  policy-server-gpu1:
    # ... config for GPU 1  
    environment:
      - CUDA_VISIBLE_DEVICES=1
    ports:
      - "50052:50051"
```

### Load Balancing

```python
import random
from roboverse_grpc.client.robosuite_policy_client import RobosuitePolicyClient

class LoadBalancedPolicyClient:
    def __init__(self, server_uris):
        self.clients = [
            RobosuitePolicyClient(RobosuitePolicyClientConfig(uri))
            for uri in server_uris
        ]
    
    def step(self, observation):
        # Simple round-robin or random selection
        client = random.choice(self.clients)
        return client.step(observation)
```

## Contributing

1. Follow the existing code structure with separate client/server/proto packages
2. Add unit tests for new conversion functions
3. Update Protocol Buffer definitions for new message types
4. Add examples for new policy wrapper types
5. Update Docker configurations for new deployment scenarios

## Troubleshooting

### Common Issues

**Connection Refused**: Ensure policy server is running and accessible
```bash
# Test connectivity
grpcurl -plaintext localhost:50051 list
```

**CUDA Out of Memory**: Reduce batch size or number of environments
```python
config = ServerConfig(batch_size=8)  # Reduce from default 16
```

**Slow Performance**: Check batching configuration and GPU utilization
```bash
nvidia-smi  # Should show high GPU utilization during inference
```

**Docker GPU Issues**: Ensure NVIDIA Container Toolkit is properly installed
```bash
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

For more detailed troubleshooting, see the [Docker README](docker/README.md).