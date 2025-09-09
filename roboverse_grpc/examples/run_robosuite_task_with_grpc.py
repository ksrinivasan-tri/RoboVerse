"""Example script to run a RoboVerse Robosuite task with gRPC policy client."""

import argparse
import time

import gymnasium as gym
from loguru import logger as log

from metasim.utils.setup_util import register_task
from roboverse_grpc.client.robosuite_policy_client import (
    RobosuitePolicyClientConfig,
)


def main():
    """Main function to run RoboVerse task with gRPC policy."""
    parser = argparse.ArgumentParser(description="Run RoboVerse Robosuite task with gRPC policy client")
    parser.add_argument("--task", default="square_d0", help="Task name (e.g., square_d0, stack_d0)")
    parser.add_argument(
        "--sim",
        default="isaaclab",
        choices=["isaacgym", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"],
        help="Simulator to use",
    )
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--policy-server", default="localhost:50051", help="Policy server URI (host:port)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )

    args = parser.parse_args()

    # Set up logging
    log.remove()  # Remove default handler
    log.add(
        lambda msg: print(msg, end=""),
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    log.info(f"Starting RoboVerse {args.task} task with gRPC policy")
    log.info(f"Simulator: {args.sim}, Environments: {args.num_envs}")
    log.info(f"Policy server: {args.policy_server}")

    try:
        # Register and create RoboVerse environment
        register_task(args.task)
        env = gym.make_vec(args.task, num_envs=args.num_envs, sim=args.sim)

        log.info(f"Created {args.num_envs} environments for task '{args.task}'")

        # Create gRPC policy client
        policy_config = RobosuitePolicyClientConfig(server_uri=args.policy_server, wait_for_server=True, timeout=30.0)
        policy_client = policy_config.create()

        log.info("Connected to gRPC policy server")

        # Get policy metadata
        try:
            metadata = policy_client.get_policy_metadata()
            log.info(f"Policy metadata: {metadata}")
        except Exception as e:
            log.warning(f"Could not get policy metadata: {e}")

        # Run episodes
        total_episodes = 0
        successful_episodes = 0
        total_steps = 0

        for episode in range(args.episodes):
            log.info(f"Starting episode {episode + 1}/{args.episodes}")

            # Reset environments
            observations, _ = env.reset()
            episode_steps = 0
            episode_rewards = [0.0] * args.num_envs
            episode_success = [False] * args.num_envs

            # Reset policy for each environment
            for env_id in range(args.num_envs):
                try:
                    policy_client.reset(env_id=env_id, seed=episode * args.num_envs + env_id)
                except Exception as e:
                    log.warning(f"Policy reset failed for env {env_id}: {e}")

            start_time = time.time()

            for step in range(args.max_steps):
                # Get actions from policy for each environment
                actions = []

                for env_id, obs in enumerate(observations):
                    try:
                        action = policy_client.step(obs, env_id=env_id)
                        actions.append(action)
                    except Exception as e:
                        log.error(f"Policy step failed for env {env_id}: {e}")
                        # Create default action
                        robot_names = list(obs.get("robots", {}).keys())
                        default_action = {}
                        for robot_name in robot_names:
                            robot_state = obs["robots"][robot_name]
                            if robot_state.get("dof_pos"):
                                joint_names = list(robot_state["dof_pos"].keys())
                                dof_pos_target = {name: robot_state["dof_pos"][name] for name in joint_names}
                                default_action[robot_name] = {
                                    "dof_pos_target": dof_pos_target,
                                    "dof_effort_target": None,
                                }
                        actions.append(default_action)

                # Step environments
                observations, rewards, successes, timeouts, _ = env.step(actions)

                episode_steps += 1
                total_steps += 1

                # Update episode tracking
                if rewards is not None:
                    for i, reward in enumerate(rewards):
                        if reward:
                            episode_rewards[i] += sum(reward)

                for i, success in enumerate(successes):
                    if success:
                        episode_success[i] = True

                # Check if any environment is done
                any_timeout = any(timeouts) if timeouts is not None else False
                any_success = any(successes) if successes is not None else False

                if any_timeout or any_success or step == args.max_steps - 1:
                    break

            episode_time = time.time() - start_time
            total_episodes += args.num_envs
            successful_episodes += sum(episode_success)

            # Log episode results
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            success_rate = sum(episode_success) / len(episode_success)

            log.info(
                f"Episode {episode + 1} completed: "
                f"{episode_steps} steps, "
                f"{episode_time:.2f}s, "
                f"avg_reward={avg_reward:.3f}, "
                f"success_rate={success_rate:.2%}"
            )

        # Final statistics
        overall_success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
        avg_steps_per_episode = total_steps / args.episodes if args.episodes > 0 else 0

        log.info("=" * 60)
        log.info("Final Results:")
        log.info(f"Total episodes: {total_episodes}")
        log.info(f"Successful episodes: {successful_episodes}")
        log.info(f"Overall success rate: {overall_success_rate:.2%}")
        log.info(f"Average steps per episode: {avg_steps_per_episode:.1f}")
        log.info(f"Total steps: {total_steps}")
        log.info("=" * 60)

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Error during execution: {e}")
        raise
    finally:
        # Clean up
        try:
            if "policy_client" in locals():
                policy_client.close()
            if "env" in locals():
                env.close()
            log.info("Cleanup completed")
        except Exception as e:
            log.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
