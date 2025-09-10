"""Simple demonstration of the RoboVerse gRPC infrastructure without external dependencies."""

import argparse
import asyncio
import sys
import time

import torch
from loguru import logger as log

from metasim.types import Action, DictEnvState
from roboverse_grpc.client.robosuite_policy_client import (
    RobosuitePolicyClientConfig,
)
from roboverse_grpc.server.policy_wrappers.base_wrapper import StatelessPolicyWrapper
from roboverse_grpc.server.robosuite_policy_server import ServerConfig, run_server


class DemoPolicyWrapper(StatelessPolicyWrapper):
    """Demo policy that generates sinusoidal joint movements."""

    def __init__(self):
        super().__init__()
        self.step_count = 0

    def step_batch(self, observations):
        """Generate sinusoidal actions for demonstration."""
        actions = {}
        self.step_count += 1

        for client_id, obs in observations.items():
            action = {}
            for robot_name, robot_state in obs.get("robots", {}).items():
                if robot_state.get("dof_pos"):
                    joint_names = list(robot_state["dof_pos"].keys())
                    dof_pos_target = {}

                    # Generate sinusoidal motion for each joint
                    for i, joint_name in enumerate(joint_names):
                        current_pos = robot_state["dof_pos"][joint_name]
                        # Different frequency and amplitude for each joint
                        amplitude = 0.1 + i * 0.05
                        frequency = 0.5 + i * 0.1
                        phase = i * 0.2

                        target = (
                            current_pos
                            + amplitude * torch.sin(torch.tensor(self.step_count * frequency + phase)).item()
                        )

                        dof_pos_target[joint_name] = target

                    action[robot_name] = {"dof_pos_target": dof_pos_target, "dof_effort_target": None}

            actions[client_id] = action

        return actions

    def get_policy_metadata(self):
        return {
            "policy_type": "sinusoidal_demo",
            "model_name": "SinusoidalDemoPolicy",
            "version": "1.0.0",
            "config": {"motion_type": "sinusoidal"},
            "supported_robots": ["franka", "ur5", "kinova"],
            "supported_tasks": ["demo"],
        }


class MockEnvironment:
    """Mock environment that simulates robot state updates."""

    def __init__(self, robot_name="franka", num_joints=7):
        self.robot_name = robot_name
        self.num_joints = num_joints
        self.joint_names = [f"joint_{i + 1}" for i in range(num_joints)]

        # Initialize joint positions
        self.joint_positions = {name: 0.0 for name in self.joint_names}
        self.joint_velocities = {name: 0.0 for name in self.joint_names}

        # Mock object and camera
        self.object_pos = torch.tensor([0.5, 0.2, 0.02])
        self.camera_data = {
            "rgb": torch.randint(0, 255, (120, 160, 3), dtype=torch.uint8),
            "depth": torch.rand(120, 160),
            "segmentation": torch.randint(0, 3, (120, 160), dtype=torch.int32),
            "intrinsics": torch.eye(3) * 100,
            "pose": torch.eye(4)[:3, :],
        }

    def get_observation(self) -> DictEnvState:
        """Get current environment observation."""
        robot_state = {
            "pos": torch.tensor([0.4, 0.0, 0.4]),
            "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            "vel": torch.tensor([0.0, 0.0, 0.0]),
            "ang_vel": torch.tensor([0.0, 0.0, 0.0]),
            "dof_pos": self.joint_positions.copy(),
            "dof_vel": self.joint_velocities.copy(),
            "com": torch.tensor([0.4, 0.0, 0.35]),
            "com_vel": torch.tensor([0.0, 0.0, 0.0]),
            "dof_pos_target": None,
            "dof_vel_target": None,
            "dof_torque": None,
        }

        object_state = {
            "pos": self.object_pos,
            "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            "vel": torch.tensor([0.0, 0.0, 0.0]),
            "ang_vel": torch.tensor([0.0, 0.0, 0.0]),
            "dof_pos": None,
            "dof_vel": None,
            "com": self.object_pos,
            "com_vel": torch.tensor([0.0, 0.0, 0.0]),
        }

        return {
            "robots": {self.robot_name: robot_state},
            "objects": {"demo_object": object_state},
            "cameras": {"demo_camera": self.camera_data},
        }

    def step(self, action: Action):
        """Update environment state based on action."""
        if self.robot_name in action:
            robot_action = action[self.robot_name]
            if robot_action.get("dof_pos_target"):
                # Simple integration: move towards target positions
                for joint_name, target_pos in robot_action["dof_pos_target"].items():
                    if joint_name in self.joint_positions:
                        current_pos = self.joint_positions[joint_name]
                        # Simple PD control simulation
                        error = target_pos - current_pos
                        new_pos = current_pos + 0.1 * error  # Simple integration
                        self.joint_positions[joint_name] = new_pos
                        self.joint_velocities[joint_name] = 0.1 * error


async def run_demo_server():
    """Run the demo policy server."""
    log.info("Starting demo policy server...")

    policy_wrapper = DemoPolicyWrapper()
    server_config = ServerConfig(host="0.0.0.0", port=50051, batch_size=4, batch_timeout_ms=20)

    try:
        await run_server(policy_wrapper, server_config)
    except KeyboardInterrupt:
        log.info("Server stopped by user")


def run_demo_client(episodes=5, steps_per_episode=20):
    """Run the demo client."""
    log.info("Starting demo client...")

    # Create policy client
    config = RobosuitePolicyClientConfig(server_uri="localhost:50051", wait_for_server=True, timeout=10.0)

    try:
        client = config.create()
        log.info("Connected to policy server")

        # Get and display policy metadata
        metadata = client.get_policy_metadata()
        log.info(f"Policy: {metadata['model_name']} ({metadata['policy_type']})")

        # Create mock environment
        env = MockEnvironment()

        for episode in range(episodes):
            log.info(f"Episode {episode + 1}/{episodes}")

            # Reset policy
            client.reset(seed=episode)

            for step in range(steps_per_episode):
                # Get observation
                obs = env.get_observation()

                # Get action from policy
                start_time = time.time()
                action = client.step(obs)
                inference_time = time.time() - start_time

                # Apply action to environment
                env.step(action)

                # Display some joint positions
                if step % 5 == 0:
                    joint_pos = list(env.joint_positions.values())[:3]
                    log.info(
                        f"  Step {step:2d}: joints=[{joint_pos[0]:6.3f}, {joint_pos[1]:6.3f}, {joint_pos[2]:6.3f}] (inference: {inference_time:.3f}s)"
                    )

            log.info(f"Episode {episode + 1} completed")

        client.close()
        log.info("Demo completed successfully!")

    except Exception as e:
        log.error(f"Demo failed: {e}")
        return False

    return True


async def main():
    """Main function to run the demonstration."""
    parser = argparse.ArgumentParser(description="RoboVerse gRPC Demo")
    parser.add_argument(
        "--mode", choices=["server", "client", "both"], default="both", help="Run server, client, or both"
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes for client")
    parser.add_argument("--steps", type=int, default=20, help="Steps per episode")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Set up logging
    log.remove()
    log.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    log.info("🤖 RoboVerse gRPC Infrastructure Demo")
    log.info("=" * 50)

    if args.mode == "server":
        # Run only server
        await run_demo_server()

    elif args.mode == "client":
        # Run only client (assumes server is already running)
        success = run_demo_client(args.episodes, args.steps)
        if not success:
            sys.exit(1)

    elif args.mode == "both":
        # Run server in background, then client
        import multiprocessing

        def server_process():
            asyncio.run(run_demo_server())

        # Start server
        server_proc = multiprocessing.Process(target=server_process)
        server_proc.start()

        try:
            # Wait for server to start
            log.info("Waiting for server to start...")
            time.sleep(2)

            # Run client
            success = run_demo_client(args.episodes, args.steps)

        finally:
            # Stop server
            log.info("Stopping server...")
            server_proc.terminate()
            server_proc.join(timeout=3)
            if server_proc.is_alive():
                server_proc.kill()
                server_proc.join()

        if not success:
            sys.exit(1)

    log.info("🎉 Demo completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Demo interrupted by user")
        sys.exit(0)
