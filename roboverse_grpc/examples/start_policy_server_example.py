"""Example script to start a policy server with different policy types."""

import argparse
import asyncio
import sys
from pathlib import Path

import torch
from loguru import logger as log

from roboverse_grpc.server.policy_wrappers.base_wrapper import StatelessPolicyWrapper
from roboverse_grpc.server.policy_wrappers.diffusion_wrapper import DiffusionPolicyWrapper
from roboverse_grpc.server.policy_wrappers.isaaclab_wrapper import IsaacLabPolicyWrapper
from roboverse_grpc.server.policy_wrappers.vla_wrapper import VLAPolicyWrapper
from roboverse_grpc.server.robosuite_policy_server import ServerConfig, run_server


class DummyDiffusionPolicy:
    """Dummy diffusion policy for testing."""

    def __init__(self, action_dim=7, device="cuda"):
        self.action_dim = action_dim
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def predict_action(self, obs_dict):
        batch_size = 1
        for key, value in obs_dict.items():
            if torch.is_tensor(value):
                batch_size = value.shape[0]
                break

        # Generate random actions
        actions = torch.randn(batch_size, self.action_dim, device=self.device)
        return {"action": actions}


class DummyVLAPolicy:
    """Dummy VLA policy for testing."""

    def __init__(self, action_dim=7, device="cuda"):
        self.action_dim = action_dim
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def predict_action(self, obs_dict):
        batch_size = 1

        # Generate random actions conditioned on language
        actions = torch.randn(batch_size, self.action_dim, device=self.device)
        return {"action": actions}


class DummyIsaacLabPolicy:
    """Dummy IsaacLab policy for testing."""

    def __init__(self, action_dim=7, device="cuda"):
        self.action_dim = action_dim
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def act(self, obs_dict):
        batch_size = 1
        for key, value in obs_dict.items():
            if torch.is_tensor(value):
                batch_size = value.shape[0]
                break

        # Generate actions in [-1, 1] range (typical for IsaacLab)
        actions = torch.tanh(torch.randn(batch_size, self.action_dim, device=self.device))
        return actions


class RandomPolicyWrapper(StatelessPolicyWrapper):
    """Simple random policy wrapper for testing."""

    def __init__(self, action_dim=7):
        super().__init__()
        self.action_dim = action_dim

    def step_batch(self, observations):
        actions = {}
        for client_id, obs in observations.items():
            action = {}
            for robot_name, robot_state in obs.get("robots", {}).items():
                if robot_state.get("dof_pos"):
                    joint_names = list(robot_state["dof_pos"].keys())
                    dof_pos_target = {}

                    # Generate random joint targets
                    for joint_name in joint_names:
                        current_pos = robot_state["dof_pos"][joint_name]
                        # Small random delta
                        delta = torch.randn(1).item() * 0.1
                        dof_pos_target[joint_name] = current_pos + delta

                    action[robot_name] = {"dof_pos_target": dof_pos_target, "dof_effort_target": None}
            actions[client_id] = action

        return actions

    def get_policy_metadata(self):
        return {
            "policy_type": "random_policy",
            "model_name": "RandomPolicy",
            "version": "1.0.0",
            "config": {"action_dim": self.action_dim},
            "supported_robots": ["any"],
            "supported_tasks": ["any"],
        }


def create_policy_wrapper(policy_type: str, model_path: str = None, device: str = "cuda"):
    """Create a policy wrapper based on the specified type."""
    if policy_type == "random":
        log.info("Creating random policy wrapper")
        return RandomPolicyWrapper()

    elif policy_type == "diffusion":
        log.info("Creating diffusion policy wrapper")
        if model_path and Path(model_path).exists():
            log.info(f"Loading diffusion model from {model_path}")
            # In a real scenario, load your trained diffusion model here
            # model = torch.load(model_path)
            model = DummyDiffusionPolicy(device=device)
        else:
            log.warning("No valid model path provided, using dummy diffusion policy")
            model = DummyDiffusionPolicy(device=device)

        return DiffusionPolicyWrapper(model, device=device)

    elif policy_type == "vla":
        log.info("Creating VLA policy wrapper")
        if model_path and Path(model_path).exists():
            log.info(f"Loading VLA model from {model_path}")
            # In a real scenario, load your trained VLA model here
            # model = load_vla_model(model_path)
            model = DummyVLAPolicy(device=device)
        else:
            log.warning("No valid model path provided, using dummy VLA policy")
            model = DummyVLAPolicy(device=device)

        return VLAPolicyWrapper(model, device=device, language_instruction="Complete the manipulation task.")

    elif policy_type == "isaaclab":
        log.info("Creating IsaacLab policy wrapper")
        if model_path and Path(model_path).exists():
            log.info(f"Loading IsaacLab model from {model_path}")
            # In a real scenario, load your trained IsaacLab model here
            # model = torch.load(model_path)
            model = DummyIsaacLabPolicy(device=device)
        else:
            log.warning("No valid model path provided, using dummy IsaacLab policy")
            model = DummyIsaacLabPolicy(device=device)

        return IsaacLabPolicyWrapper(model, device=device)

    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


async def main():
    """Main function to start the policy server."""
    parser = argparse.ArgumentParser(description="Start RoboVerse gRPC Policy Server")
    parser.add_argument(
        "--policy-type",
        choices=["random", "diffusion", "vla", "isaaclab"],
        default="random",
        help="Type of policy to serve",
    )
    parser.add_argument("--model-path", type=str, help="Path to trained model file")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for policy inference"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Maximum batch size for processing")
    parser.add_argument("--batch-timeout-ms", type=int, default=50, help="Batch timeout in milliseconds")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum worker threads")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )

    args = parser.parse_args()

    # Set up logging
    log.remove()
    log.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    log.info("Starting RoboVerse gRPC Policy Server")
    log.info(f"Policy type: {args.policy_type}")
    log.info(f"Device: {args.device}")
    log.info(f"Server address: {args.host}:{args.port}")

    try:
        # Check device availability
        if args.device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available, falling back to CPU")
            args.device = "cpu"

        if args.device == "cuda":
            log.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            log.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Create policy wrapper
        policy_wrapper = create_policy_wrapper(args.policy_type, args.model_path, args.device)

        # Create server configuration
        server_config = ServerConfig(
            host=args.host,
            port=args.port,
            batch_size=args.batch_size,
            batch_timeout_ms=args.batch_timeout_ms,
            max_workers=args.max_workers,
        )

        log.info(f"Server configuration: {server_config}")

        # Start server
        await run_server(policy_wrapper, server_config)

    except KeyboardInterrupt:
        log.info("Server interrupted by user")
    except Exception as e:
        log.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    # Check for required dependencies
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required. Please install with: pip install torch")
        sys.exit(1)

    try:
        import grpc
    except ImportError:
        print("Error: gRPC is required. Please install with: pip install grpcio grpcio-tools")
        sys.exit(1)

    # Run the server
    asyncio.run(main())
