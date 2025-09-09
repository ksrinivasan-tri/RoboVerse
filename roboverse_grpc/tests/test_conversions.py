"""Unit tests for RoboVerse gRPC conversion functions."""

import unittest
import uuid

import torch

from metasim.types import Action, EnvState, ObjectState, RobotState
from roboverse_grpc.conversions.robosuite_policy_conversions import (
    action_to_grpc_msg,
    env_state_to_grpc_msg,
    grpc_msg_to_action,
    grpc_msg_to_env_state,
    grpc_msg_to_policy_metadata,
    grpc_msg_to_uuid,
    grpc_tensor_to_tensor,
    policy_metadata_to_grpc_msg,
    tensor_to_grpc_tensor,
    uuid_to_grpc_msg,
)
from roboverse_grpc.proto import RobosuitePolicy_pb2


class TestTensorConversions(unittest.TestCase):
    """Test tensor conversion functions."""

    def test_tensor_conversion_basic(self):
        """Test basic tensor conversion."""
        # Test 1D tensor
        original_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        grpc_tensor = tensor_to_grpc_tensor(original_tensor)
        converted_tensor = grpc_tensor_to_tensor(grpc_tensor)

        self.assertTrue(torch.allclose(original_tensor, converted_tensor))
        self.assertEqual(list(grpc_tensor.shape), [4])
        self.assertEqual(grpc_tensor.dtype, str(original_tensor.dtype).replace("torch.", ""))

    def test_tensor_conversion_multidimensional(self):
        """Test multidimensional tensor conversion."""
        # Test 3D tensor
        original_tensor = torch.randn(2, 3, 4)
        grpc_tensor = tensor_to_grpc_tensor(original_tensor)
        converted_tensor = grpc_tensor_to_tensor(grpc_tensor)

        self.assertTrue(torch.allclose(original_tensor, converted_tensor))
        self.assertEqual(list(grpc_tensor.shape), [2, 3, 4])

    def test_tensor_conversion_empty(self):
        """Test empty tensor conversion."""
        grpc_tensor = RobosuitePolicy_pb2.Tensor()
        converted_tensor = grpc_tensor_to_tensor(grpc_tensor)

        self.assertEqual(converted_tensor.numel(), 0)

    def test_tensor_conversion_none(self):
        """Test None tensor conversion."""
        grpc_tensor = tensor_to_grpc_tensor(None)
        self.assertEqual(len(grpc_tensor.data), 0)


class TestStateConversions(unittest.TestCase):
    """Test state conversion functions."""

    def create_sample_env_state(self) -> EnvState:
        """Create a sample environment state for testing."""
        robot_state: RobotState = {
            "pos": torch.tensor([0.5, 0.0, 0.3]),
            "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),  # quaternion
            "vel": torch.tensor([0.1, 0.0, 0.0]),
            "ang_vel": torch.tensor([0.0, 0.0, 0.1]),
            "dof_pos": {"joint1": 0.5, "joint2": -0.3, "joint3": 0.8},
            "dof_vel": {"joint1": 0.1, "joint2": 0.0, "joint3": -0.2},
            "com": torch.tensor([0.5, 0.0, 0.25]),
            "com_vel": torch.tensor([0.1, 0.0, 0.0]),
            "dof_pos_target": {"joint1": 0.6, "joint2": -0.2, "joint3": 0.9},
            "dof_vel_target": None,
            "dof_torque": {"joint1": 0.1, "joint2": 0.05, "joint3": -0.03},
        }

        object_state: ObjectState = {
            "pos": torch.tensor([0.7, 0.2, 0.1]),
            "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            "vel": torch.tensor([0.0, 0.0, 0.0]),
            "ang_vel": torch.tensor([0.0, 0.0, 0.0]),
            "dof_pos": None,
            "dof_vel": None,
            "com": torch.tensor([0.7, 0.2, 0.1]),
            "com_vel": torch.tensor([0.0, 0.0, 0.0]),
        }

        camera_data = {
            "rgb": torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8),
            "depth": torch.rand(480, 640),
            "segmentation": torch.randint(0, 10, (480, 640), dtype=torch.int32),
            "intrinsics": torch.tensor([500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]).reshape(3, 3),
            "pose": torch.tensor([1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]).reshape(3, 4),
        }

        env_state: EnvState = {
            "robots": {"franka": robot_state},
            "objects": {"cube": object_state},
            "cameras": {"front_cam": camera_data},
        }

        return env_state

    def test_env_state_conversion(self):
        """Test environment state conversion."""
        original_env_state = self.create_sample_env_state()

        # Convert to gRPC and back
        grpc_obs = env_state_to_grpc_msg(original_env_state, env_id=5)
        converted_env_state, env_id = grpc_msg_to_env_state(grpc_obs)

        # Check env_id
        self.assertEqual(env_id, 5)

        # Check robots
        self.assertIn("franka", converted_env_state["robots"])
        converted_robot = converted_env_state["robots"]["franka"]
        original_robot = original_env_state["robots"]["franka"]

        self.assertTrue(torch.allclose(converted_robot["pos"], original_robot["pos"]))
        self.assertTrue(torch.allclose(converted_robot["rot"], original_robot["rot"]))
        self.assertEqual(converted_robot["dof_pos"], original_robot["dof_pos"])

        # Check objects
        self.assertIn("cube", converted_env_state["objects"])
        converted_object = converted_env_state["objects"]["cube"]
        original_object = original_env_state["objects"]["cube"]

        self.assertTrue(torch.allclose(converted_object["pos"], original_object["pos"]))

        # Check cameras
        self.assertIn("front_cam", converted_env_state["cameras"])
        converted_camera = converted_env_state["cameras"]["front_cam"]
        original_camera = original_env_state["cameras"]["front_cam"]

        self.assertTrue(torch.allclose(converted_camera["rgb"], original_camera["rgb"]))
        self.assertTrue(torch.allclose(converted_camera["depth"], original_camera["depth"]))


class TestActionConversions(unittest.TestCase):
    """Test action conversion functions."""

    def test_action_conversion(self):
        """Test action conversion."""
        original_action: Action = {
            "franka": {
                "dof_pos_target": {"joint1": 0.5, "joint2": -0.3, "joint3": 0.8},
                "dof_effort_target": {"joint1": 0.1, "joint2": 0.05, "joint3": -0.03},
            },
            "ur5": {"dof_pos_target": {"shoulder_pan": 1.0, "shoulder_lift": -0.5}, "dof_effort_target": None},
        }

        # Convert to gRPC and back
        grpc_action = action_to_grpc_msg(original_action, env_id=3)
        converted_action, env_id = grpc_msg_to_action(grpc_action)

        # Check env_id
        self.assertEqual(env_id, 3)

        # Check action structure
        self.assertIn("franka", converted_action)
        self.assertIn("ur5", converted_action)

        # Check franka action
        franka_action = converted_action["franka"]
        self.assertEqual(franka_action["dof_pos_target"], original_action["franka"]["dof_pos_target"])
        self.assertEqual(franka_action["dof_effort_target"], original_action["franka"]["dof_effort_target"])

        # Check ur5 action
        ur5_action = converted_action["ur5"]
        self.assertEqual(ur5_action["dof_pos_target"], original_action["ur5"]["dof_pos_target"])
        self.assertIsNone(ur5_action["dof_effort_target"])


class TestUtilityConversions(unittest.TestCase):
    """Test utility conversion functions."""

    def test_uuid_conversion(self):
        """Test UUID conversion."""
        original_uuid = uuid.uuid4()
        grpc_string = uuid_to_grpc_msg(original_uuid)
        converted_uuid = grpc_msg_to_uuid(grpc_string)

        self.assertEqual(original_uuid, converted_uuid)
        self.assertIsInstance(grpc_string, str)

    def test_policy_metadata_conversion(self):
        """Test policy metadata conversion."""
        original_metadata = {
            "policy_type": "diffusion_policy",
            "model_name": "DiffusionTransformer",
            "version": "2.1.0",
            "config": {"batch_size": "16", "device": "cuda"},
            "supported_robots": ["franka", "ur5", "kinova"],
            "supported_tasks": ["pick_and_place", "reaching"],
        }

        # Convert to gRPC and back
        grpc_metadata = policy_metadata_to_grpc_msg(**original_metadata)
        converted_metadata = grpc_msg_to_policy_metadata(grpc_metadata)

        # Check all fields
        for key, value in original_metadata.items():
            self.assertEqual(converted_metadata[key], value)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_env_state(self):
        """Test conversion of empty environment state."""
        empty_env_state: EnvState = {"robots": {}, "objects": {}, "cameras": {}}

        grpc_obs = env_state_to_grpc_msg(empty_env_state)
        converted_env_state, env_id = grpc_msg_to_env_state(grpc_obs)

        self.assertEqual(len(converted_env_state["robots"]), 0)
        self.assertEqual(len(converted_env_state["objects"]), 0)
        self.assertEqual(len(converted_env_state["cameras"]), 0)

    def test_empty_action(self):
        """Test conversion of empty action."""
        empty_action: Action = {}

        grpc_action = action_to_grpc_msg(empty_action)
        converted_action, env_id = grpc_msg_to_action(grpc_action)

        self.assertEqual(len(converted_action), 0)

    def test_partial_robot_state(self):
        """Test conversion of partial robot state."""
        partial_robot_state: RobotState = {
            "pos": torch.tensor([0.5, 0.0, 0.3]),
            "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            "vel": None,
            "ang_vel": None,
            "dof_pos": {"joint1": 0.5},
            "dof_vel": None,
            "com": None,
            "com_vel": None,
            "dof_pos_target": None,
            "dof_vel_target": None,
            "dof_torque": None,
        }

        env_state: EnvState = {"robots": {"test_robot": partial_robot_state}, "objects": {}, "cameras": {}}

        grpc_obs = env_state_to_grpc_msg(env_state)
        converted_env_state, env_id = grpc_msg_to_env_state(grpc_obs)

        converted_robot = converted_env_state["robots"]["test_robot"]
        self.assertTrue(torch.allclose(converted_robot["pos"], partial_robot_state["pos"]))
        self.assertEqual(converted_robot["dof_pos"], {"joint1": 0.5})


if __name__ == "__main__":
    unittest.main()
