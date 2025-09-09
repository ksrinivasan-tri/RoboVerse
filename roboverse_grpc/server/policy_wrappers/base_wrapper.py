"""Base policy wrapper interface for RoboVerse gRPC server."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from uuid import UUID

from metasim.types import Action, EnvState


class PolicyWrapper(ABC):
    """Abstract base class for policy wrappers.

    Policy wrappers adapt different policy implementations (IsaacLab, diffusion models,
    VLA models, etc.) to work with the gRPC server infrastructure.
    """

    @abstractmethod
    def step_batch(self, observations: Dict[UUID, EnvState]) -> Dict[UUID, Action]:
        """Process a batch of observations and return corresponding actions.

        Args:
            observations: Dictionary mapping client UUIDs to environment observations.

        Returns:
            Dictionary mapping client UUIDs to policy actions.
        """
        pass

    @abstractmethod
    def reset_batch(self, client_seeds: Dict[UUID, Optional[int]]):
        """Reset policy state for a batch of clients.

        Args:
            client_seeds: Dictionary mapping client UUIDs to optional random seeds.
        """
        pass

    @abstractmethod
    def get_policy_metadata(self) -> Dict[str, Any]:
        """Get metadata describing this policy.

        Returns:
            Dictionary containing policy type, model name, version, etc.
        """
        pass

    def initialize(self):
        """Initialize the policy wrapper (called once at server startup)."""
        pass

    def shutdown(self):
        """Clean up policy wrapper resources (called at server shutdown)."""
        pass


class StatelessPolicyWrapper(PolicyWrapper):
    """Base class for stateless policies that don't need reset functionality."""

    def reset_batch(self, client_seeds: Dict[UUID, Optional[int]]):
        """No-op reset for stateless policies."""
        pass


class SingleInstancePolicyWrapper(PolicyWrapper):
    """Base class for policies that maintain a single model instance for all clients."""

    def __init__(self, policy_instance):
        """Initialize with a single policy instance.

        Args:
            policy_instance: The underlying policy model/function.
        """
        self.policy = policy_instance

    def step_batch(self, observations: Dict[UUID, EnvState]) -> Dict[UUID, Action]:
        """Process observations one by one using the single policy instance.

        This is a simple implementation that processes each observation individually.
        Subclasses can override this to implement true batch processing if supported.
        """
        actions = {}
        for client_id, observation in observations.items():
            try:
                action = self._step_single(observation)
                actions[client_id] = action
            except Exception as e:
                # Log error but continue processing other observations
                print(f"Error processing observation for client {client_id}: {e}")
                # Return a default/safe action or skip this client
                actions[client_id] = self._get_default_action()
        return actions

    @abstractmethod
    def _step_single(self, observation: EnvState) -> Action:
        """Process a single observation.

        Args:
            observation: Single environment state observation.

        Returns:
            Single action for the observation.
        """
        pass

    def _get_default_action(self) -> Action:
        """Get a default/safe action when policy fails.

        Returns:
            A safe default action (e.g., all zeros).
        """
        # Return empty action - subclasses should override for meaningful defaults
        return {}

    def reset_batch(self, client_seeds: Dict[UUID, Optional[int]]):
        """Reset the single policy instance.

        For single-instance policies, we typically reset once for any reset request.
        """
        if hasattr(self.policy, "reset"):
            # Use first available seed if any
            seed = next(iter(client_seeds.values())) if client_seeds else None
            self.policy.reset(seed=seed)


class MultiInstancePolicyWrapper(PolicyWrapper):
    """Base class for policies that maintain separate instances per client."""

    def __init__(self):
        """Initialize with empty client policy mapping."""
        self.client_policies: Dict[UUID, Any] = {}

    def step_batch(self, observations: Dict[UUID, EnvState]) -> Dict[UUID, Action]:
        """Process observations using client-specific policy instances."""
        actions = {}
        for client_id, observation in observations.items():
            try:
                # Get or create policy instance for this client
                if client_id not in self.client_policies:
                    self.client_policies[client_id] = self._create_policy_instance(client_id)

                policy = self.client_policies[client_id]
                action = self._step_single_with_policy(policy, observation)
                actions[client_id] = action
            except Exception as e:
                print(f"Error processing observation for client {client_id}: {e}")
                actions[client_id] = self._get_default_action()
        return actions

    @abstractmethod
    def _create_policy_instance(self, client_id: UUID) -> Any:
        """Create a new policy instance for a client.

        Args:
            client_id: UUID of the client requesting a policy instance.

        Returns:
            New policy instance for the client.
        """
        pass

    @abstractmethod
    def _step_single_with_policy(self, policy: Any, observation: EnvState) -> Action:
        """Process a single observation with a specific policy instance.

        Args:
            policy: The policy instance to use.
            observation: Single environment state observation.

        Returns:
            Single action for the observation.
        """
        pass

    def _get_default_action(self) -> Action:
        """Get a default/safe action when policy fails."""
        return {}

    def reset_batch(self, client_seeds: Dict[UUID, Optional[int]]):
        """Reset policy instances for specified clients."""
        for client_id, seed in client_seeds.items():
            if client_id in self.client_policies:
                policy = self.client_policies[client_id]
                if hasattr(policy, "reset"):
                    policy.reset(seed=seed)

    def shutdown(self):
        """Clean up all client policy instances."""
        for policy in self.client_policies.values():
            if hasattr(policy, "close"):
                policy.close()
        self.client_policies.clear()
