"""IsaacLab policy wrapper for RoboVerse gRPC server."""

from typing import Any, Dict, List

import numpy as np
import torch

from metasim.types import Action, DictEnvState

from .base_wrapper import SingleInstancePolicyWrapper


class IsaacLabPolicyWrapper(SingleInstancePolicyWrapper):
    """Wrapper for IsaacLab/IsaacGym-trained policies compatible with RoboVerse observations."""

    def __init__(self, policy_model, device: str = "cuda", obs_keys: List[str] = None, normalize_obs: bool = True):
        """Initialize IsaacLab policy wrapper.

        Args:
            policy_model: Trained IsaacLab policy model (e.g., from RSL-RL).
            device: Device to run inference on ("cuda" or "cpu").
            obs_keys: List of observation keys expected by the policy.
            normalize_obs: Whether to apply observation normalization.
        """
        super().__init__(policy_model)
        self.device = device
        self.obs_keys = obs_keys or ["policy"]  # Default observation key
        self.normalize_obs = normalize_obs

        # Move model to device
        if hasattr(self.policy, "to"):
            self.policy.to(device)

        # Set to evaluation mode
        if hasattr(self.policy, "eval"):
            self.policy.eval()

        # Initialize observation and action normalization parameters
        self.obs_mean = None
        self.obs_std = None
        self.action_mean = None
        self.action_std = None

        # Try to extract normalization parameters from model
        self._extract_normalization_params()

    def _extract_normalization_params(self):
        """Extract observation and action normalization parameters from the policy."""
        if hasattr(self.policy, "obs_normalizer"):
            normalizer = self.policy.obs_normalizer
            if hasattr(normalizer, "mean") and hasattr(normalizer, "std"):
                self.obs_mean = normalizer.mean
                self.obs_std = normalizer.std

        if hasattr(self.policy, "action_normalizer"):
            normalizer = self.policy.action_normalizer
            if hasattr(normalizer, "mean") and hasattr(normalizer, "std"):
                self.action_mean = normalizer.mean
                self.action_std = normalizer.std

    def _step_single(self, observation: DictEnvState) -> Action:
        """Process single observation through IsaacLab policy.

        Args:
            observation: RoboVerse environment state.

        Returns:
            Action from IsaacLab policy.
        """
        # Convert observation to IsaacLab format
        policy_obs = self._convert_observation(observation)

        with torch.no_grad():
            # Get action from policy
            if hasattr(self.policy, "act"):
                # RSL-RL style policy
                actions = self.policy.act(policy_obs)
            elif hasattr(self.policy, "predict"):
                # Stable-Baselines3 style policy
                actions, _ = self.policy.predict(policy_obs)
            elif hasattr(self.policy, "forward"):
                # Direct PyTorch model
                actions = self.policy.forward(policy_obs)
            elif callable(self.policy):
                # Direct callable
                actions = self.policy(policy_obs)
            else:
                raise ValueError("IsaacLab policy must have act, predict, forward method, or be callable")

            # Denormalize actions if needed
            if self.action_mean is not None and self.action_std is not None:
                actions = actions * self.action_std + self.action_mean

            # Convert actions back to RoboVerse format
            return self._convert_action(actions, observation)

    def _convert_observation(self, observation: DictEnvState) -> Dict[str, torch.Tensor]:
        """Convert RoboVerse observation to IsaacLab policy format.

        Args:
            observation: RoboVerse environment state.

        Returns:
            Dictionary of tensors expected by IsaacLab policy.
        """
        obs_dict = {}
        obs_list = []

        # Process robot observations
        for robot_name, robot_state in observation.get("robots", {}).items():
            robot_obs = []

            # Joint positions
            if robot_state.get("dof_pos"):
                joint_positions = list(robot_state["dof_pos"].values())
                robot_obs.extend(joint_positions)

            # Joint velocities
            if robot_state.get("dof_vel"):
                joint_velocities = list(robot_state["dof_vel"].values())
                robot_obs.extend(joint_velocities)

            # End-effector position
            if robot_state.get("pos") is not None:
                ee_pos = robot_state["pos"].flatten().tolist()
                robot_obs.extend(ee_pos)

            # End-effector orientation (quaternion or rotation matrix)
            if robot_state.get("rot") is not None:
                ee_rot = robot_state["rot"].flatten().tolist()
                robot_obs.extend(ee_rot)

            # Linear velocity
            if robot_state.get("vel") is not None:
                lin_vel = robot_state["vel"].flatten().tolist()
                robot_obs.extend(lin_vel)

            # Angular velocity
            if robot_state.get("ang_vel") is not None:
                ang_vel = robot_state["ang_vel"].flatten().tolist()
                robot_obs.extend(ang_vel)

            obs_list.extend(robot_obs)

        # Process object observations (if relevant for the policy)
        for obj_name, obj_state in observation.get("objects", {}).items():
            obj_obs = []

            # Object position
            if obj_state.get("pos") is not None:
                obj_pos = obj_state["pos"].flatten().tolist()
                obj_obs.extend(obj_pos)

            # Object orientation
            if obj_state.get("rot") is not None:
                obj_rot = obj_state["rot"].flatten().tolist()
                obj_obs.extend(obj_rot)

            # Object velocity (if available)
            if obj_state.get("vel") is not None:
                obj_vel = obj_state["vel"].flatten().tolist()
                obj_obs.extend(obj_vel)

            obs_list.extend(obj_obs)

        # Convert to tensor
        if obs_list:
            obs_tensor = torch.tensor(obs_list, dtype=torch.float32, device=self.device)
            obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension

            # Apply normalization if available
            if self.normalize_obs and self.obs_mean is not None and self.obs_std is not None:
                obs_tensor = (obs_tensor - self.obs_mean) / (self.obs_std + 1e-8)

            # Map to expected observation keys
            for key in self.obs_keys:
                obs_dict[key] = obs_tensor

        return obs_dict

    def _convert_action(self, actions: torch.Tensor, observation: DictEnvState) -> Action:
        """Convert IsaacLab policy action to RoboVerse format.

        Args:
            actions: Action tensor from IsaacLab policy.
            observation: Original observation for context.

        Returns:
            RoboVerse action dictionary.
        """
        action_dict: Action = {}

        # Convert tensor to numpy
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy().squeeze()
        else:
            actions_np = np.array(actions).squeeze()

        # Ensure 1D array
        if actions_np.ndim == 0:
            actions_np = np.array([actions_np])

        # Map actions to robots based on observation structure
        robot_names = list(observation.get("robots", {}).keys())

        if len(robot_names) == 1:
            robot_name = robot_names[0]
            robot_state = observation["robots"][robot_name]

            # Map actions to joint targets
            if robot_state.get("dof_pos"):
                joint_names = list(robot_state["dof_pos"].keys())
                dof_pos_target = {}

                # Map action values to joint names
                for i, joint_name in enumerate(joint_names):
                    if i < len(actions_np):
                        # For IsaacLab, actions are often joint position deltas or direct targets
                        current_pos = robot_state["dof_pos"][joint_name]

                        # Determine if action is delta or absolute
                        # This heuristic assumes actions in [-1, 1] are deltas
                        if abs(actions_np[i]) <= 1.0:
                            # Delta action - scale and add to current position
                            action_scale = 0.1  # Scaling factor for delta actions
                            target_pos = current_pos + actions_np[i] * action_scale
                        else:
                            # Absolute action
                            target_pos = actions_np[i]

                        dof_pos_target[joint_name] = float(target_pos)

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
                            current_pos = robot_state["dof_pos"][joint_name]

                            if abs(robot_actions[j]) <= 1.0:
                                action_scale = 0.1
                                target_pos = current_pos + robot_actions[j] * action_scale
                            else:
                                target_pos = robot_actions[j]

                            dof_pos_target[joint_name] = float(target_pos)

                    action_dict[robot_name] = {"dof_pos_target": dof_pos_target, "dof_effort_target": None}

        return action_dict

    def _get_default_action(self) -> Action:
        """Get default safe action for IsaacLab policy."""
        return {"default_robot": {"dof_pos_target": {}, "dof_effort_target": None}}

    def get_policy_metadata(self) -> Dict[str, Any]:
        """Get IsaacLab policy metadata."""
        return {
            "policy_type": "isaaclab_policy",
            "model_name": getattr(self.policy, "__class__", {}).get("__name__", "IsaacLabPolicy"),
            "version": "1.0.0",
            "config": {
                "device": self.device,
                "obs_keys": self.obs_keys,
                "normalize_obs": self.normalize_obs,
                "model_type": "neural_network",
            },
            "supported_robots": ["franka", "ur5", "kinova", "anymal", "quadruped"],
            "supported_tasks": ["locomotion", "manipulation", "reaching", "tracking"],
        }

    def set_obs_keys(self, obs_keys: List[str]):
        """Update observation keys expected by the policy.

        Args:
            obs_keys: List of observation keys.
        """
        self.obs_keys = obs_keys

    def initialize(self):
        """Initialize IsaacLab policy."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def shutdown(self):
        """Clean up IsaacLab policy resources."""
        if hasattr(self.policy, "close"):
            self.policy.close()
