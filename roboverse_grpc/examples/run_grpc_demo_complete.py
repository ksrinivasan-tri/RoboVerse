"""Complete working example of RoboVerse gRPC infrastructure.
This script demonstrates both policy server and environment client without requiring external assets.
"""

import argparse
import asyncio
import multiprocessing
import time

import numpy as np
import torch
from loguru import logger as log

from metasim.types import EnvState, ObjectState, RobotState
from roboverse_grpc.client.robosuite_policy_client import (
    RobosuitePolicyClientConfig,
)
from roboverse_grpc.server.policy_wrappers.base_wrapper import StatelessPolicyWrapper
from roboverse_grpc.server.robosuite_policy_server import ServerConfig, run_server


class DemoEnvironment:
    """Mock environment that simulates RoboVerse-style observations and actions."""

    def __init__(self, num_envs=4, robot_name="franka", task_name="square_d0"):
        self.num_envs = num_envs
        self.robot_name = robot_name
        self.task_name = task_name

        # Joint configuration for Franka robot
        self.joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        # Initialize robot states for each environment
        self.robot_states = []
        for env_id in range(num_envs):
            initial_positions = {name: np.random.uniform(-0.5, 0.5) for name in self.joint_names}
            self.robot_states.append(initial_positions)

        # Object states
        self.object_positions = [np.array([0.5 + 0.1 * i, 0.2, 0.02]) for i in range(num_envs)]

        # Episode tracking
        self.episode_steps = [0] * num_envs
        self.max_episode_steps = 100

    def reset(self):
        """Reset all environments and return initial observations."""
        log.info(f"Resetting {self.num_envs} demo environments")

        # Reset episode steps
        self.episode_steps = [0] * self.num_envs

        # Reset robot states to initial positions
        for env_id in range(self.num_envs):
            self.robot_states[env_id] = {name: np.random.uniform(-0.3, 0.3) for name in self.joint_names}

        observations = self.get_observations()
        return observations, {}

    def get_observations(self):
        """Get current observations for all environments."""
        observations = []

        for env_id in range(self.num_envs):
            # Robot state
            robot_state: RobotState = {
                "pos": torch.tensor([0.4, 0.0, 0.4]),  # End-effector position
                "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),  # Quaternion
                "vel": torch.tensor([0.0, 0.0, 0.0]),
                "ang_vel": torch.tensor([0.0, 0.0, 0.0]),
                "dof_pos": self.robot_states[env_id],
                "dof_vel": {name: 0.0 for name in self.joint_names},
                "com": torch.tensor([0.4, 0.0, 0.35]),
                "com_vel": torch.tensor([0.0, 0.0, 0.0]),
                "dof_pos_target": None,
                "dof_vel_target": None,
                "dof_torque": None,
            }

            # Object state (e.g., cube to manipulate)
            object_state: ObjectState = {
                "pos": torch.from_numpy(self.object_positions[env_id]).float(),
                "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "vel": torch.tensor([0.0, 0.0, 0.0]),
                "ang_vel": torch.tensor([0.0, 0.0, 0.0]),
                "dof_pos": None,
                "dof_vel": None,
                "com": torch.from_numpy(self.object_positions[env_id]).float(),
                "com_vel": torch.tensor([0.0, 0.0, 0.0]),
            }

            # Camera data (simulated RGB image)
            camera_data = {
                "rgb": torch.randint(0, 255, (240, 320, 3), dtype=torch.uint8),
                "depth": torch.rand(240, 320),
                "segmentation": torch.randint(0, 5, (240, 320), dtype=torch.int32),
                "intrinsics": torch.tensor([200.0, 0.0, 160.0, 0.0, 200.0, 120.0, 0.0, 0.0, 1.0]).reshape(3, 3),
                "pose": torch.eye(4)[:3, :],
            }

            env_state: EnvState = {
                "robots": {self.robot_name: robot_state},
                "objects": {"target_cube": object_state},
                "cameras": {"front_camera": camera_data},
            }

            observations.append(env_state)

        return observations

    def step(self, actions):
        """Execute actions and return next observations."""
        observations = []
        rewards = []
        successes = []
        timeouts = []

        for env_id in range(self.num_envs):
            # Update robot state based on action
            if env_id < len(actions):
                action = actions[env_id]
                if self.robot_name in action:
                    robot_action = action[self.robot_name]
                    if robot_action.get("dof_pos_target"):
                        # Simple simulation: move towards target positions
                        for joint_name, target_pos in robot_action["dof_pos_target"].items():
                            if joint_name in self.robot_states[env_id]:
                                current_pos = self.robot_states[env_id][joint_name]
                                # Simple PD-like control
                                error = target_pos - current_pos
                                new_pos = current_pos + 0.1 * error
                                self.robot_states[env_id][joint_name] = new_pos

            # Increment episode step
            self.episode_steps[env_id] += 1

            # Simple reward: negative distance to target
            robot_ee_pos = np.array([0.4, 0.0, 0.4])  # Simplified EE position
            target_pos = self.object_positions[env_id]
            distance = np.linalg.norm(robot_ee_pos - target_pos)
            reward = -distance
            rewards.append([reward])  # RoboVerse expects nested list

            # Success condition: close to target
            success = distance < 0.1
            successes.append(success)

            # Timeout condition
            timeout = self.episode_steps[env_id] >= self.max_episode_steps
            timeouts.append(timeout)

        # Get updated observations
        observations = self.get_observations()

        return observations, rewards, successes, timeouts, {}


class SinusoidalPolicyWrapper(StatelessPolicyWrapper):
    """Demo policy that generates smooth sinusoidal joint movements."""

    def __init__(self):
        super().__init__()
        self.step_count = 0

    def step_batch(self, observations):
        """Generate smooth sinusoidal actions for all observations."""
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
                        amplitude = 0.05 + i * 0.02
                        frequency = 0.3 + i * 0.1
                        phase = i * np.pi / 4

                        # Sinusoidal target with offset from current position
                        offset = amplitude * np.sin(self.step_count * frequency + phase)
                        target = current_pos + offset

                        # Clamp to reasonable joint limits
                        target = np.clip(target, -1.5, 1.5)
                        dof_pos_target[joint_name] = float(target)

                    action[robot_name] = {"dof_pos_target": dof_pos_target, "dof_effort_target": None}

            actions[client_id] = action

        return actions

    def get_policy_metadata(self):
        return {
            "policy_type": "sinusoidal_demo",
            "model_name": "SinusoidalDemoPolicy",
            "version": "1.0.0",
            "config": {"motion_type": "sinusoidal", "smooth": True},
            "supported_robots": ["franka", "ur5", "kinova"],
            "supported_tasks": ["demo", "reaching", "manipulation"],
        }


async def run_policy_server_async(port=50051):
    """Run the gRPC policy server asynchronously."""
    log.info(f"Starting demo policy server on port {port}...")

    policy_wrapper = SinusoidalPolicyWrapper()
    server_config = ServerConfig(
        host="0.0.0.0",
        port=port,
        batch_size=16,
        batch_timeout_ms=20,  # Low latency for demo
    )

    await run_server(policy_wrapper, server_config)


def run_policy_server_process(port=50051):
    """Run policy server in a separate process."""
    asyncio.run(run_policy_server_async(port))


def run_environment_client(server_uri="localhost:50051", num_envs=4, episodes=5, max_steps_per_episode=50):
    """Run the environment client that connects to the policy server."""
    log.info(f"Starting environment client with {num_envs} environments")
    log.info(f"Connecting to policy server at {server_uri}")

    # Create policy client
    config = RobosuitePolicyClientConfig(server_uri=server_uri, wait_for_server=True, timeout=10.0)

    try:
        client = config.create()
        log.info("Successfully connected to policy server")

        # Get policy metadata
        try:
            metadata = client.get_policy_metadata()
            log.info(f"Policy: {metadata['model_name']} ({metadata['policy_type']})")
        except Exception as e:
            log.warning(f"Could not get policy metadata: {e}")

        # Create demo environment
        env = DemoEnvironment(num_envs=num_envs)

        # Run episodes
        total_steps = 0
        successful_episodes = 0

        for episode in range(episodes):
            log.info(f"Episode {episode + 1}/{episodes}")

            # Reset environment
            observations, _ = env.reset()

            # Reset policy for each environment
            for env_id in range(num_envs):
                try:
                    client.reset(env_id=env_id, seed=episode * num_envs + env_id)
                except Exception as e:
                    log.warning(f"Policy reset failed for env {env_id}: {e}")

            episode_rewards = [0.0] * num_envs
            episode_steps = 0

            for step in range(max_steps_per_episode):
                # Get actions from policy
                actions = []

                start_time = time.time()
                for env_id, obs in enumerate(observations):
                    try:
                        action = client.step(obs, env_id=env_id)
                        actions.append(action)
                    except Exception as e:
                        log.error(f"Policy step failed for env {env_id}: {e}")
                        # Create safe default action
                        default_action = {env.robot_name: {"dof_pos_target": {}, "dof_effort_target": None}}
                        actions.append(default_action)

                inference_time = time.time() - start_time

                # Execute actions in environment
                observations, rewards, successes, timeouts, _ = env.step(actions)

                # Update tracking
                episode_steps += 1
                total_steps += 1

                if rewards:
                    for i, reward_list in enumerate(rewards):
                        if reward_list:
                            episode_rewards[i] += reward_list[0]

                # Log progress
                if step % 10 == 0:
                    avg_reward = sum(episode_rewards) / len(episode_rewards)
                    success_count = sum(successes)
                    log.info(
                        f"  Step {step:2d}: avg_reward={avg_reward:6.3f}, "
                        f"successes={success_count}/{num_envs}, "
                        f"inference_time={inference_time:.3f}s"
                    )

                # Check if any environment finished
                if any(timeouts) or any(successes):
                    break

            # Episode summary
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            success_count = sum(successes)
            successful_episodes += success_count

            log.info(
                f"Episode {episode + 1} completed: "
                f"{episode_steps} steps, "
                f"avg_reward={avg_reward:.3f}, "
                f"successes={success_count}/{num_envs}"
            )

        # Final summary
        overall_success_rate = successful_episodes / (episodes * num_envs)
        avg_steps_per_episode = total_steps / episodes

        log.info("=" * 60)
        log.info("Demo Results:")
        log.info(f"Total episodes: {episodes * num_envs}")
        log.info(f"Successful episodes: {successful_episodes}")
        log.info(f"Success rate: {overall_success_rate:.2%}")
        log.info(f"Average steps per episode: {avg_steps_per_episode:.1f}")
        log.info(f"Total steps: {total_steps}")
        log.info("=" * 60)

        client.close()
        return True

    except Exception as e:
        log.error(f"Environment client failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to run the complete demo."""
    parser = argparse.ArgumentParser(description="RoboVerse gRPC Complete Demo")
    parser.add_argument(
        "--mode", choices=["server", "client", "both"], default="both", help="Run server, client, or both"
    )
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Set up logging
    log.remove()
    log.add(
        lambda msg: print(msg, end=""),
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    log.info("🚀 RoboVerse gRPC Complete Demo")
    log.info("=" * 60)

    if args.mode == "server":
        # Run only server
        try:
            asyncio.run(run_policy_server_async(args.port))
        except KeyboardInterrupt:
            log.info("Server stopped by user")

    elif args.mode == "client":
        # Run only client (assumes server is already running)
        success = run_environment_client(
            server_uri=f"localhost:{args.port}",
            num_envs=args.num_envs,
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
        )
        if not success:
            return 1

    elif args.mode == "both":
        # Run server in background, then client
        log.info("Starting in combined mode...")

        # Start server process
        server_proc = multiprocessing.Process(target=run_policy_server_process, args=(args.port,))
        server_proc.start()

        try:
            # Wait for server to start
            log.info("Waiting for server to start...")
            time.sleep(3)

            # Run client
            success = run_environment_client(
                server_uri=f"localhost:{args.port}",
                num_envs=args.num_envs,
                episodes=args.episodes,
                max_steps_per_episode=args.max_steps,
            )

        finally:
            # Stop server
            log.info("Stopping server...")
            server_proc.terminate()
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                server_proc.kill()
                server_proc.join()

        if not success:
            return 1

    log.info("🎉 Demo completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
