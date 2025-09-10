"""Test script to validate the gRPC infrastructure without requiring external assets."""

import asyncio
import multiprocessing
import time
import uuid
from typing import Dict

import torch
from loguru import logger as log

from metasim.types import Action, DictEnvState, ObjectState, RobotState
from roboverse_grpc.client.robosuite_policy_client import (
    RobosuitePolicyClientConfig,
)
from roboverse_grpc.server.policy_wrappers.base_wrapper import StatelessPolicyWrapper
from roboverse_grpc.server.robosuite_policy_server import ServerConfig, run_server


class MockPolicyWrapper(StatelessPolicyWrapper):
    """Mock policy wrapper for testing."""

    def step_batch(self, observations: Dict[uuid.UUID, DictEnvState]) -> Dict[uuid.UUID, Action]:
        """Generate mock actions for all observations."""
        actions = {}
        for client_id, obs in observations.items():
            action = {}
            for robot_name, robot_state in obs.get("robots", {}).items():
                if robot_state.get("dof_pos"):
                    joint_names = list(robot_state["dof_pos"].keys())
                    # Generate small random deltas for joint positions
                    dof_pos_target = {}
                    for joint_name in joint_names:
                        current_pos = robot_state["dof_pos"][joint_name]
                        delta = torch.randn(1).item() * 0.05  # Small random delta
                        dof_pos_target[joint_name] = current_pos + delta

                    action[robot_name] = {"dof_pos_target": dof_pos_target, "dof_effort_target": None}
            actions[client_id] = action
        return actions

    def get_policy_metadata(self):
        return {
            "policy_type": "mock_policy",
            "model_name": "MockTestPolicy",
            "version": "1.0.0",
            "config": {"type": "test"},
            "supported_robots": ["franka", "ur5"],
            "supported_tasks": ["any"],
        }


def create_mock_env_state(env_id: int = 0) -> DictEnvState:
    """Create a mock environment state for testing."""
    # Mock robot state
    robot_state: RobotState = {
        "pos": torch.tensor([0.5, 0.0, 0.3]),
        "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),  # quaternion
        "vel": torch.tensor([0.0, 0.0, 0.0]),
        "ang_vel": torch.tensor([0.0, 0.0, 0.0]),
        "dof_pos": {
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,
            "panda_joint7": 0.785,
        },
        "dof_vel": {
            "panda_joint1": 0.0,
            "panda_joint2": 0.0,
            "panda_joint3": 0.0,
            "panda_joint4": 0.0,
            "panda_joint5": 0.0,
            "panda_joint6": 0.0,
            "panda_joint7": 0.0,
        },
        "com": torch.tensor([0.5, 0.0, 0.25]),
        "com_vel": torch.tensor([0.0, 0.0, 0.0]),
        "dof_pos_target": None,
        "dof_vel_target": None,
        "dof_torque": None,
    }

    # Mock object state
    object_state: ObjectState = {
        "pos": torch.tensor([0.6, 0.1, 0.02]),
        "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "vel": torch.tensor([0.0, 0.0, 0.0]),
        "ang_vel": torch.tensor([0.0, 0.0, 0.0]),
        "dof_pos": None,
        "dof_vel": None,
        "com": torch.tensor([0.6, 0.1, 0.02]),
        "com_vel": torch.tensor([0.0, 0.0, 0.0]),
    }

    # Mock camera data
    camera_data = {
        "rgb": torch.randint(0, 255, (240, 320, 3), dtype=torch.uint8),
        "depth": torch.rand(240, 320),
        "segmentation": torch.randint(0, 5, (240, 320), dtype=torch.int32),
        "intrinsics": torch.tensor([200.0, 0.0, 160.0, 0.0, 200.0, 120.0, 0.0, 0.0, 1.0]).reshape(3, 3),
        "pose": torch.eye(4)[:3, :],  # 3x4 transformation matrix
    }

    env_state: DictEnvState = {
        "robots": {"franka": robot_state},
        "objects": {"cube": object_state},
        "cameras": {"front_camera": camera_data},
    }

    return env_state


async def run_policy_server():
    """Run the policy server in a separate process."""
    log.info("Starting mock policy server...")

    policy_wrapper = MockPolicyWrapper()
    server_config = ServerConfig(host="0.0.0.0", port=50051, batch_size=8, batch_timeout_ms=50)

    await run_server(policy_wrapper, server_config)


def start_server_process():
    """Start the server in a separate process."""
    asyncio.run(run_policy_server())


def test_client_server_communication():
    """Test the gRPC client-server communication."""
    log.info("Testing gRPC client-server communication...")

    # Start server in background process
    server_process = multiprocessing.Process(target=start_server_process)
    server_process.start()

    try:
        # Wait for server to start
        time.sleep(3)

        # Create client
        config = RobosuitePolicyClientConfig(server_uri="localhost:50051", wait_for_server=True, timeout=10.0)
        client = config.create()

        log.info("Successfully connected to policy server")

        # Test get metadata
        try:
            metadata = client.get_policy_metadata()
            log.info(f"Policy metadata: {metadata}")
        except Exception as e:
            log.error(f"Failed to get metadata: {e}")
            return False

        # Test reset
        try:
            client.reset(seed=42, env_id=0)
            log.info("Policy reset successful")
        except Exception as e:
            log.error(f"Failed to reset policy: {e}")
            return False

        # Test multiple step operations
        success_count = 0
        total_tests = 10

        for i in range(total_tests):
            try:
                # Create mock observation
                obs = create_mock_env_state(env_id=i)

                # Get action from policy
                start_time = time.time()
                action = client.step(obs, env_id=i)
                inference_time = time.time() - start_time

                # Validate action structure
                if "franka" in action:
                    franka_action = action["franka"]
                    if franka_action.get("dof_pos_target"):
                        success_count += 1
                        log.info(f"Step {i + 1}/{total_tests}: Success (inference: {inference_time:.3f}s)")
                    else:
                        log.warning(f"Step {i + 1}/{total_tests}: Invalid action structure")
                else:
                    log.warning(f"Step {i + 1}/{total_tests}: No franka action in response")

            except Exception as e:
                log.error(f"Step {i + 1}/{total_tests}: Failed - {e}")

        # Test batch processing
        try:
            log.info("Testing batch processing...")
            batch_observations = [create_mock_env_state(i) for i in range(5)]
            batch_env_ids = list(range(5))

            start_time = time.time()
            batch_actions = client.step_batch(batch_observations, batch_env_ids)
            batch_time = time.time() - start_time

            log.info(f"Batch processing: {len(batch_actions)} actions in {batch_time:.3f}s")

        except Exception as e:
            log.error(f"Batch processing failed: {e}")

        # Results
        success_rate = success_count / total_tests
        log.info(f"Test Results: {success_count}/{total_tests} successful ({success_rate:.1%})")

        # Cleanup
        client.close()

        return success_rate > 0.8  # 80% success rate threshold

    finally:
        # Stop server
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
            server_process.join()


def test_conversion_roundtrip():
    """Test that observation/action conversions are lossless."""
    log.info("Testing conversion round-trip...")

    from roboverse_grpc.conversions.robosuite_policy_conversions import (
        action_to_grpc_msg,
        env_state_to_grpc_msg,
        grpc_msg_to_action,
        grpc_msg_to_env_state,
    )

    # Test observation conversion
    original_obs = create_mock_env_state()
    grpc_obs = env_state_to_grpc_msg(original_obs, env_id=5)
    converted_obs, env_id = grpc_msg_to_env_state(grpc_obs)

    assert env_id == 5, f"Environment ID mismatch: {env_id} != 5"

    # Check robot state
    original_robot = original_obs["robots"]["franka"]
    converted_robot = converted_obs["robots"]["franka"]

    assert torch.allclose(original_robot["pos"], converted_robot["pos"]), "Robot position mismatch"
    # Compare DOF positions with tolerance for floating point precision
    for joint_name in original_robot["dof_pos"]:
        assert abs(original_robot["dof_pos"][joint_name] - converted_robot["dof_pos"][joint_name]) < 1e-6, (
            f"Robot DOF position mismatch for {joint_name}"
        )

    # Test action conversion
    original_action: Action = {
        "franka": {
            "dof_pos_target": {
                "panda_joint1": 0.1,
                "panda_joint2": -0.8,
                "panda_joint3": 0.05,
            },
            "dof_effort_target": None,
        }
    }

    grpc_action = action_to_grpc_msg(original_action, env_id=3)
    converted_action, action_env_id = grpc_msg_to_action(grpc_action)

    assert action_env_id == 3, f"Action environment ID mismatch: {action_env_id} != 3"
    assert converted_action == original_action, "Action conversion mismatch"

    log.info("Conversion round-trip test passed!")
    return True


def main():
    """Main test function."""
    log.remove()
    log.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    log.info("🚀 Starting RoboVerse gRPC Infrastructure Test")
    log.info("=" * 60)

    all_tests_passed = True

    # Test 1: Conversion functions
    try:
        log.info("Test 1: Conversion round-trip")
        conversion_success = test_conversion_roundtrip()
        log.info(f"✅ Conversion test: {'PASSED' if conversion_success else 'FAILED'}")
        all_tests_passed &= conversion_success
    except Exception as e:
        log.error(f"❌ Conversion test FAILED: {e}")
        all_tests_passed = False

    log.info("-" * 60)

    # Test 2: Client-server communication
    try:
        log.info("Test 2: Client-server communication")
        communication_success = test_client_server_communication()
        log.info(f"✅ Communication test: {'PASSED' if communication_success else 'FAILED'}")
        all_tests_passed &= communication_success
    except Exception as e:
        log.error(f"❌ Communication test FAILED: {e}")
        all_tests_passed = False

    log.info("=" * 60)

    if all_tests_passed:
        log.info("🎉 All tests PASSED! gRPC infrastructure is working correctly.")
    else:
        log.error("💥 Some tests FAILED. Please check the implementation.")

    log.info("🏁 Test completed")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit(main())
