"""Vision-Language-Action (VLA) policy wrapper for RoboVerse gRPC server."""

from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from metasim.types import Action, EnvState

from .base_wrapper import SingleInstancePolicyWrapper


class VLAPolicyWrapper(SingleInstancePolicyWrapper):
    """Wrapper for Vision-Language-Action policies compatible with RoboVerse observations."""

    def __init__(
        self,
        policy_model,
        device: str = "cuda",
        image_size: tuple = (224, 224),
        language_instruction: Optional[str] = None,
    ):
        """Initialize VLA policy wrapper.

        Args:
            policy_model: Trained VLA policy model.
            device: Device to run inference on ("cuda" or "cpu").
            image_size: Target image size for VLA model (width, height).
            language_instruction: Default language instruction if not provided in observation.
        """
        super().__init__(policy_model)
        self.device = device
        self.image_size = image_size
        self.default_instruction = language_instruction or "Complete the manipulation task."

        # Move model to device
        if hasattr(self.policy, "to"):
            self.policy.to(device)

        # Set to evaluation mode
        if hasattr(self.policy, "eval"):
            self.policy.eval()

    def _step_single(self, observation: EnvState) -> Action:
        """Process single observation through VLA policy.

        Args:
            observation: RoboVerse environment state.

        Returns:
            Action from VLA policy.
        """
        # Convert observation to VLA format
        vla_input = self._convert_observation(observation)

        with torch.no_grad():
            # Get action from VLA policy
            if hasattr(self.policy, "predict_action"):
                result = self.policy.predict_action(vla_input)
                actions = result.get("action", result)
            elif hasattr(self.policy, "step"):
                actions = self.policy.step(vla_input)
            elif callable(self.policy):
                actions = self.policy(vla_input)
            else:
                raise ValueError("VLA policy must have predict_action, step method, or be callable")

            # Convert actions back to RoboVerse format
            return self._convert_action(actions, observation)

    def _convert_observation(self, observation: EnvState) -> Dict[str, Any]:
        """Convert RoboVerse observation to VLA policy format.

        Args:
            observation: RoboVerse environment state.

        Returns:
            Dictionary with images, robot state, and language instruction.
        """
        vla_obs = {}

        # Process camera observations
        images = []
        for camera_name, camera_data in observation.get("cameras", {}).items():
            if camera_data.get("rgb") is not None:
                rgb_tensor = camera_data["rgb"]

                # Convert tensor to PIL Image
                if isinstance(rgb_tensor, torch.Tensor):
                    # Ensure correct format (H, W, C) and range [0, 255]
                    if rgb_tensor.max() <= 1.0:
                        rgb_tensor = rgb_tensor * 255
                    rgb_np = rgb_tensor.detach().cpu().numpy().astype(np.uint8)

                    if rgb_np.shape[-1] == 3:  # (H, W, C)
                        image = Image.fromarray(rgb_np)
                    elif rgb_np.shape[0] == 3:  # (C, H, W)
                        image = Image.fromarray(rgb_np.transpose(1, 2, 0))
                    else:
                        raise ValueError(f"Unexpected RGB image shape: {rgb_np.shape}")

                    # Resize to target size
                    image = image.resize(self.image_size, Image.BILINEAR)
                    images.append(image)

        if images:
            vla_obs["images"] = images
            # Primary image for single-image VLA models
            vla_obs["image"] = images[0]

        # Process robot state (proprioception)
        robot_states = []
        for robot_name, robot_state in observation.get("robots", {}).items():
            state_dict = {}

            # Joint positions
            if robot_state.get("dof_pos"):
                joint_positions = list(robot_state["dof_pos"].values())
                state_dict["joint_positions"] = torch.tensor(joint_positions, dtype=torch.float32, device=self.device)

            # Joint velocities
            if robot_state.get("dof_vel"):
                joint_velocities = list(robot_state["dof_vel"].values())
                state_dict["joint_velocities"] = torch.tensor(joint_velocities, dtype=torch.float32, device=self.device)

            # End-effector pose
            if robot_state.get("pos") is not None:
                state_dict["ee_position"] = robot_state["pos"].to(self.device)

            if robot_state.get("rot") is not None:
                state_dict["ee_orientation"] = robot_state["rot"].to(self.device)

            robot_states.append(state_dict)

        if robot_states:
            vla_obs["robot_state"] = robot_states[0]  # Primary robot
            vla_obs["robot_states"] = robot_states  # All robots

            # Flatten robot state for compatibility
            flattened_state = []
            for state in robot_states:
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        flattened_state.extend(value.flatten().tolist())
                    elif isinstance(value, (list, tuple)):
                        flattened_state.extend(value)

            vla_obs["proprioception"] = torch.tensor(flattened_state, dtype=torch.float32, device=self.device)

        # Language instruction
        vla_obs["instruction"] = self.default_instruction
        vla_obs["language"] = self.default_instruction
        vla_obs["text"] = self.default_instruction

        return vla_obs

    def _convert_action(self, actions: Any, observation: EnvState) -> Action:
        """Convert VLA policy action to RoboVerse format.

        Args:
            actions: Action from VLA policy (could be tensor, dict, or other format).
            observation: Original observation for context.

        Returns:
            RoboVerse action dictionary.
        """
        action_dict: Action = {}

        # Handle different action formats
        if isinstance(actions, dict):
            # Action is already in dictionary format
            if "action" in actions:
                actions = actions["action"]
            elif "actions" in actions:
                actions = actions["actions"]

        # Convert to numpy array
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        elif isinstance(actions, (list, tuple)):
            actions_np = np.array(actions)
        else:
            actions_np = actions

        # Ensure 1D array
        if actions_np.ndim > 1:
            actions_np = actions_np.flatten()

        # Map actions to robots
        robot_names = list(observation.get("robots", {}).keys())

        if len(robot_names) == 1:
            robot_name = robot_names[0]
            robot_state = observation["robots"][robot_name]

            # Map actions to joint positions
            if robot_state.get("dof_pos"):
                joint_names = list(robot_state["dof_pos"].keys())
                dof_pos_target = {}

                # Handle different action interpretations
                if len(actions_np) == len(joint_names):
                    # Direct joint position commands
                    for i, joint_name in enumerate(joint_names):
                        dof_pos_target[joint_name] = float(actions_np[i])
                elif len(actions_np) == len(joint_names) * 2:
                    # Joint positions + gripper (take first N for positions)
                    for i, joint_name in enumerate(joint_names):
                        if i < len(joint_names):
                            dof_pos_target[joint_name] = float(actions_np[i])
                elif len(actions_np) >= 6:
                    # End-effector pose commands - convert to joint space
                    # This would typically require inverse kinematics
                    # For now, map first N actions to joints
                    for i, joint_name in enumerate(joint_names):
                        if i < len(actions_np):
                            dof_pos_target[joint_name] = float(actions_np[i])
                else:
                    # Fallback: replicate action across all joints
                    action_value = float(actions_np[0]) if len(actions_np) > 0 else 0.0
                    for joint_name in joint_names:
                        dof_pos_target[joint_name] = action_value

                action_dict[robot_name] = {"dof_pos_target": dof_pos_target, "dof_effort_target": None}
        else:
            # Multi-robot case
            actions_per_robot = len(actions_np) // len(robot_names)

            for i, robot_name in enumerate(robot_names):
                robot_state = observation["robots"][robot_name]
                start_idx = i * actions_per_robot
                end_idx = (i + 1) * actions_per_robot
                robot_actions = actions_np[start_idx:end_idx]

                if robot_state.get("dof_pos"):
                    joint_names = list(robot_state["dof_pos"].keys())
                    dof_pos_target = {}

                    for j, joint_name in enumerate(joint_names):
                        if j < len(robot_actions):
                            dof_pos_target[joint_name] = float(robot_actions[j])

                    action_dict[robot_name] = {"dof_pos_target": dof_pos_target, "dof_effort_target": None}

        return action_dict

    def _get_default_action(self) -> Action:
        """Get default safe action for VLA policy."""
        return {"default_robot": {"dof_pos_target": {}, "dof_effort_target": None}}

    def get_policy_metadata(self) -> Dict[str, Any]:
        """Get VLA policy metadata."""
        return {
            "policy_type": "vision_language_action",
            "model_name": getattr(self.policy, "__class__", {}).get("__name__", "VLAPolicy"),
            "version": "1.0.0",
            "config": {
                "device": self.device,
                "image_size": self.image_size,
                "model_type": "vla",
                "multimodal": True,
                "language_conditioned": True,
            },
            "supported_robots": ["franka", "ur5", "kinova", "fetch"],
            "supported_tasks": ["manipulation", "navigation", "pick_and_place", "language_conditioned"],
        }

    def set_language_instruction(self, instruction: str):
        """Update the default language instruction.

        Args:
            instruction: New language instruction for the policy.
        """
        self.default_instruction = instruction

    def initialize(self):
        """Initialize VLA policy."""
        if hasattr(self.policy, "initialize"):
            self.policy.initialize()

    def shutdown(self):
        """Clean up VLA policy resources."""
        if hasattr(self.policy, "close"):
            self.policy.close()
