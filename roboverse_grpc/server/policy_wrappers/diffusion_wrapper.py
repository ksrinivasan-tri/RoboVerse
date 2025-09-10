"""Diffusion policy wrapper for RoboVerse gRPC server."""

from typing import Any, Dict

import torch

from metasim.types import Action, DictEnvState

from .base_wrapper import SingleInstancePolicyWrapper


class DiffusionPolicyWrapper(SingleInstancePolicyWrapper):
    """Wrapper for diffusion policies compatible with RoboVerse observations."""

    def __init__(self, policy_model, device: str = "cuda"):
        """Initialize diffusion policy wrapper.

        Args:
            policy_model: Trained diffusion policy model.
            device: Device to run inference on ("cuda" or "cpu").
        """
        super().__init__(policy_model)
        self.device = device
        self.policy.to(device) if hasattr(self.policy, "to") else None

        # Set model to evaluation mode
        if hasattr(self.policy, "eval"):
            self.policy.eval()

    def _step_single(self, observation: DictEnvState) -> Action:
        """Process single observation through diffusion policy.

        Args:
            observation: RoboVerse environment state.

        Returns:
            Action from diffusion policy.
        """
        # Convert RoboVerse observation to format expected by diffusion policy
        obs_dict = self._convert_observation(observation)

        with torch.no_grad():
            # Get action from diffusion policy
            if hasattr(self.policy, "predict_action"):
                # Standard diffusion policy interface
                result = self.policy.predict_action(obs_dict)
                actions = result["action"]
            elif callable(self.policy):
                # Direct callable policy
                actions = self.policy(obs_dict)
            else:
                raise ValueError("Policy must have predict_action method or be callable")

            # Convert actions back to RoboVerse format
            return self._convert_action(actions, observation)

    def _convert_observation(self, observation: DictEnvState) -> Dict[str, torch.Tensor]:
        """Convert RoboVerse observation to diffusion policy format.

        Args:
            observation: RoboVerse environment state.

        Returns:
            Dictionary of tensors expected by diffusion policy.
        """
        obs_dict = {}

        # Handle robot state
        for robot_name, robot_state in observation.get("robots", {}).items():
            # Joint positions
            if robot_state.get("dof_pos"):
                joint_positions = torch.tensor(
                    list(robot_state["dof_pos"].values()), dtype=torch.float32, device=self.device
                ).unsqueeze(0)  # Add batch dimension
                obs_dict[f"{robot_name}_joint_pos"] = joint_positions

            # End-effector pose
            if robot_state.get("pos") is not None:
                ee_pos = robot_state["pos"].to(self.device).unsqueeze(0)
                obs_dict[f"{robot_name}_ee_pos"] = ee_pos

            if robot_state.get("rot") is not None:
                ee_rot = robot_state["rot"].to(self.device).unsqueeze(0)
                obs_dict[f"{robot_name}_ee_rot"] = ee_rot

        # Handle camera observations
        for camera_name, camera_data in observation.get("cameras", {}).items():
            if camera_data.get("rgb") is not None:
                # Ensure proper image format (B, C, H, W)
                rgb_image = camera_data["rgb"].to(self.device)
                if rgb_image.dim() == 3:  # (H, W, C)
                    rgb_image = rgb_image.permute(2, 0, 1)  # (C, H, W)
                rgb_image = rgb_image.unsqueeze(0)  # (B, C, H, W)
                obs_dict[f"{camera_name}_rgb"] = rgb_image

            if camera_data.get("depth") is not None:
                depth_image = camera_data["depth"].to(self.device)
                if depth_image.dim() == 2:  # (H, W)
                    depth_image = depth_image.unsqueeze(0)  # (1, H, W)
                depth_image = depth_image.unsqueeze(0)  # (B, 1, H, W)
                obs_dict[f"{camera_name}_depth"] = depth_image

        return obs_dict

    def _convert_action(self, actions: torch.Tensor, observation: DictEnvState) -> Action:
        """Convert diffusion policy action to RoboVerse format.

        Args:
            actions: Action tensor from diffusion policy.
            observation: Original observation for context.

        Returns:
            RoboVerse action dictionary.
        """
        action_dict: Action = {}

        # Convert tensor to numpy for processing
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy().squeeze()
        else:
            actions_np = actions

        # Map actions to robots based on observation structure
        robot_names = list(observation.get("robots", {}).keys())

        if len(robot_names) == 1:
            robot_name = robot_names[0]
            robot_state = observation["robots"][robot_name]

            # Get joint names to map actions
            if robot_state.get("dof_pos"):
                joint_names = list(robot_state["dof_pos"].keys())

                # Map action values to joint names
                dof_pos_target = {}
                for i, joint_name in enumerate(joint_names):
                    if i < len(actions_np):
                        dof_pos_target[joint_name] = float(actions_np[i])

                action_dict[robot_name] = {"dof_pos_target": dof_pos_target, "dof_effort_target": None}
        else:
            # Handle multi-robot case - split actions appropriately
            action_idx = 0
            for robot_name in robot_names:
                robot_state = observation["robots"][robot_name]
                if robot_state.get("dof_pos"):
                    joint_names = list(robot_state["dof_pos"].keys())
                    dof_pos_target = {}

                    for joint_name in joint_names:
                        if action_idx < len(actions_np):
                            dof_pos_target[joint_name] = float(actions_np[action_idx])
                            action_idx += 1

                    action_dict[robot_name] = {"dof_pos_target": dof_pos_target, "dof_effort_target": None}

        return action_dict

    def _get_default_action(self) -> Action:
        """Get default safe action (all zeros)."""
        return {"default_robot": {"dof_pos_target": {}, "dof_effort_target": None}}

    def get_policy_metadata(self) -> Dict[str, Any]:
        """Get diffusion policy metadata."""
        return {
            "policy_type": "diffusion_policy",
            "model_name": getattr(self.policy, "__class__", {}).get("__name__", "DiffusionPolicy"),
            "version": "1.0.0",
            "config": {"device": self.device, "model_type": "diffusion"},
            "supported_robots": ["franka", "ur5", "kinova"],
            "supported_tasks": ["manipulation", "reaching", "grasping"],
        }

    def initialize(self):
        """Initialize diffusion policy."""
        if hasattr(self.policy, "initialize"):
            self.policy.initialize()

    def shutdown(self):
        """Clean up diffusion policy resources."""
        if hasattr(self.policy, "close"):
            self.policy.close()
