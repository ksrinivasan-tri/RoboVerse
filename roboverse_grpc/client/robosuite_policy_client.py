"""gRPC client for RoboVerse policies that integrates with the metasim environment."""

import dataclasses as dc
import uuid
from typing import Any, Dict, List, Optional

import grpc
from loguru import logger as log

from metasim.types import Action, DictEnvState
from roboverse_grpc.conversions.robosuite_policy_conversions import (
    env_state_to_grpc_msg,
    grpc_msg_to_action,
    grpc_msg_to_policy_metadata,
    uuid_to_grpc_msg,
)
from roboverse_grpc.proto import RobosuitePolicy_pb2, RobosuitePolicy_pb2_grpc


@dc.dataclass
class RobosuitePolicyClientConfig:
    """Configuration for RoboVerse gRPC policy client.

    Args:
        wait_for_server (bool): Whether to block initialization until server is ready.
        server_uri (str): Server address in 'host:port' format.
        grpc_max_send_message_length (int): Maximum message size for sending.
        grpc_max_receive_message_length (int): Maximum message size for receiving.
        timeout (float): Request timeout in seconds.
        policy_type (str): Type of policy being used.
    """

    wait_for_server: bool = True
    server_uri: str = "localhost:50051"
    grpc_max_send_message_length: int = 50 * 1024 * 1024  # 50 MB
    grpc_max_receive_message_length: int = 50 * 1024 * 1024  # 50 MB
    timeout: float = 30.0  # 30 seconds
    policy_type: str = "robosuite_policy"

    def create(self):
        """Create a policy client instance."""
        return RobosuitePolicyClient(config=self)


class RobosuitePolicyClient:
    """A policy client that communicates with a gRPC policy server for RoboVerse environments.

    This client converts RoboVerse DictEnvState observations into gRPC messages,
    sends them to a remote policy server, and converts the received actions
    back into RoboVerse Action format.
    """

    def __init__(self, config: RobosuitePolicyClientConfig):
        assert isinstance(config, RobosuitePolicyClientConfig)

        self._config = config
        self._server_uri = config.server_uri
        self._timeout = config.timeout

        # gRPC channel options
        self._grpc_options = [
            ("grpc.max_send_message_length", config.grpc_max_send_message_length),
            ("grpc.max_receive_message_length", config.grpc_max_receive_message_length),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000),
        ]

        # Generate unique client identifier
        self._uuid = uuid.uuid4()
        self._client_identifier = uuid_to_grpc_msg(self._uuid)

        log.info(f"Initializing RoboVerse gRPC policy client with UUID: {self._uuid}")

        # Wait for server if configured
        if config.wait_for_server:
            self._wait_for_server()

    def _wait_for_server(self):
        """Wait for the gRPC server to be ready."""
        log.info(f"Waiting for gRPC policy server at {self._server_uri}")

        with grpc.insecure_channel(self._server_uri, options=self._grpc_options) as channel:
            # Wait for the channel to be ready
            try:
                grpc.channel_ready_future(channel).result(timeout=self._timeout)
                log.info("Successfully connected to gRPC policy server")
            except grpc.FutureTimeoutError:
                raise RuntimeError(
                    f"Failed to connect to gRPC server at {self._server_uri} within {self._timeout} seconds"
                )

    def _success_or_throw(self, response, operation: str = "operation"):
        """Check response success and raise error if failed."""
        if not response.success:
            error_msg = getattr(response, "error_message", "Unknown error")
            raise RuntimeError(f"Server rejected {operation} for client {self._uuid}: {error_msg}")

    def get_policy_metadata(self) -> Dict[str, Any]:
        """Get metadata about the policy from the server."""
        request = RobosuitePolicy_pb2.GetPolicyMetadataRequest(client_identifier=self._client_identifier)

        try:
            with grpc.insecure_channel(self._server_uri, options=self._grpc_options) as channel:
                stub = RobosuitePolicy_pb2_grpc.RobosuiteGetPolicyMetadataServiceStub(channel)
                response = stub.GetPolicyMetadata(request, timeout=self._timeout, wait_for_ready=True)
        except grpc.RpcError as e:
            log.error(f"gRPC error in get_policy_metadata: {e}")
            raise RuntimeError(f"Failed to get policy metadata: {e}")

        self._success_or_throw(response, "get_policy_metadata")
        return grpc_msg_to_policy_metadata(response.policy_metadata)

    def reset(self, seed: Optional[int] = None, env_id: int = 0, options: Optional[Dict[str, Any]] = None):
        """Reset the policy for a specific environment.

        Args:
            seed: Random seed for reproducibility.
            env_id: Environment identifier for multi-environment setups.
            options: Additional reset options (currently unused).
        """
        request = RobosuitePolicy_pb2.RobosuitePolicyResetRequest(
            client_identifier=self._client_identifier, env_id=env_id
        )

        if seed is not None:
            request.seed = seed

        try:
            with grpc.insecure_channel(self._server_uri, options=self._grpc_options) as channel:
                stub = RobosuitePolicy_pb2_grpc.RobosuitePolicyResetServiceStub(channel)
                response = stub.PolicyReset(request, timeout=self._timeout, wait_for_ready=True)
        except grpc.RpcError as e:
            log.error(f"gRPC error in reset: {e}")
            raise RuntimeError(f"Failed to reset policy: {e}")

        self._success_or_throw(response, "reset")
        log.debug(f"Successfully reset policy for environment {env_id}")

    def step(self, observation: DictEnvState, env_id: int = 0) -> Action:
        """Get action from policy given an observation.

        Args:
            observation: RoboVerse environment state observation.
            env_id: Environment identifier for multi-environment setups.

        Returns:
            Action to be executed in the environment.
        """
        # Convert observation to gRPC message
        grpc_observation = env_state_to_grpc_msg(observation, env_id)

        request = RobosuitePolicy_pb2.RobosuitePolicyStepRequest(
            client_identifier=self._client_identifier, observation=grpc_observation
        )

        try:
            with grpc.insecure_channel(self._server_uri, options=self._grpc_options) as channel:
                stub = RobosuitePolicy_pb2_grpc.RobosuitePolicyStepServiceStub(channel)
                response = stub.PolicyStep(request, timeout=self._timeout, wait_for_ready=True)
        except grpc.RpcError as e:
            log.error(f"gRPC error in step: {e}")
            raise RuntimeError(f"Failed to get policy action: {e}")

        self._success_or_throw(response, "step")

        # Convert action back to RoboVerse format
        action, returned_env_id = grpc_msg_to_action(response.action)

        if returned_env_id != env_id:
            log.warning(f"Environment ID mismatch: sent {env_id}, received {returned_env_id}")

        return action

    def step_batch(self, observations: List[DictEnvState], env_ids: Optional[List[int]] = None) -> List[Action]:
        """Get actions for a batch of observations.

        Args:
            observations: List of RoboVerse environment state observations.
            env_ids: Environment identifiers. If None, uses sequential IDs starting from 0.

        Returns:
            List of actions corresponding to the input observations.
        """
        if env_ids is None:
            env_ids = list(range(len(observations)))

        if len(observations) != len(env_ids):
            raise ValueError(f"Observations and env_ids length mismatch: {len(observations)} != {len(env_ids)}")

        # For now, process each observation individually
        # TODO: Implement true batch processing on server side
        actions = []
        for observation, env_id in zip(observations, env_ids):
            action = self.step(observation, env_id)
            actions.append(action)

        return actions

    def close(self):
        """Clean up client resources."""
        log.info(f"Closing RoboVerse gRPC policy client {self._uuid}")
        # No persistent connections to close in current implementation

    @property
    def client_id(self) -> str:
        """Get the client identifier."""
        return self._client_identifier

    @property
    def server_uri(self) -> str:
        """Get the server URI."""
        return self._server_uri


class RoboVersePolicyWrapper:
    """Wrapper to make the gRPC client compatible with existing RoboVerse policy interfaces."""

    def __init__(self, client: RobosuitePolicyClient):
        self.client = client
        self._metadata = None

    def __call__(self, observation: DictEnvState) -> Action:
        """Make the wrapper callable like a policy function."""
        return self.client.step(observation)

    def step(self, observation: DictEnvState) -> Action:
        """Step method for policy interface compatibility."""
        return self.client.step(observation)

    def reset(self, **kwargs):
        """Reset method for policy interface compatibility."""
        return self.client.reset(**kwargs)

    def get_metadata(self) -> Dict[str, Any]:
        """Get policy metadata with caching."""
        if self._metadata is None:
            self._metadata = self.client.get_policy_metadata()
        return self._metadata

    def close(self):
        """Close the underlying client."""
        self.client.close()
