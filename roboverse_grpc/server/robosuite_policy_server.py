"""gRPC server for RoboVerse policies that can serve multiple policy types."""

import asyncio
import dataclasses as dc
import signal
import time
from concurrent import futures
from typing import Dict
from uuid import UUID

from grpc import aio
from loguru import logger as log

from roboverse_grpc.conversions.robosuite_policy_conversions import (
    action_to_grpc_msg,
    grpc_msg_to_env_state,
    grpc_msg_to_uuid,
    policy_metadata_to_grpc_msg,
)
from roboverse_grpc.proto import RobosuitePolicy_pb2, RobosuitePolicy_pb2_grpc
from roboverse_grpc.server.policy_wrappers.base_wrapper import PolicyWrapper


@dc.dataclass
class ServerConfig:
    """Configuration for the RoboVerse gRPC policy server.

    Args:
        host: Host address to bind the server to.
        port: Port number to bind the server to.
        max_workers: Maximum number of worker threads.
        max_message_length: Maximum message size for gRPC.
        batch_size: Maximum batch size for processing observations.
        batch_timeout_ms: Timeout for batching observations in milliseconds.
        enable_reflection: Whether to enable gRPC reflection for debugging.
        enable_health_check: Whether to enable gRPC health checking.
    """

    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_length: int = 50 * 1024 * 1024  # 50 MB
    batch_size: int = 32
    batch_timeout_ms: int = 50  # 50ms timeout for batching
    enable_reflection: bool = True
    enable_health_check: bool = True


class RobosuitePolicyServer:
    """Async gRPC server for RoboVerse policies with batching support."""

    def __init__(self, policy_wrapper: PolicyWrapper, config: ServerConfig):
        """Initialize the policy server.

        Args:
            policy_wrapper: Wrapper around the policy implementation.
            config: Server configuration.
        """
        self.policy_wrapper = policy_wrapper
        self.config = config
        self.server = None

        # Batching state
        self.pending_requests: Dict[UUID, tuple] = {}
        self.batch_processing_task = None
        self.shutdown_event = asyncio.Event()

        log.info(f"Initializing RoboVerse gRPC policy server on {config.host}:{config.port}")

    async def start(self):
        """Start the gRPC server."""
        # Initialize policy wrapper
        self.policy_wrapper.initialize()

        # Create gRPC server
        self.server = aio.server(
            futures.ThreadPoolExecutor(max_workers=self.config.max_workers),
            options=[
                ("grpc.max_send_message_length", self.config.max_message_length),
                ("grpc.max_receive_message_length", self.config.max_message_length),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
            ],
        )

        # Add service implementations
        RobosuitePolicy_pb2_grpc.add_RobosuitePolicyStepServiceServicer_to_server(PolicyStepServicer(self), self.server)
        RobosuitePolicy_pb2_grpc.add_RobosuitePolicyResetServiceServicer_to_server(
            PolicyResetServicer(self), self.server
        )
        RobosuitePolicy_pb2_grpc.add_RobosuiteGetPolicyMetadataServiceServicer_to_server(
            GetPolicyMetadataServicer(self), self.server
        )

        if self.config.enable_health_check:
            RobosuitePolicy_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), self.server)

        # Enable reflection if requested
        if self.config.enable_reflection:
            from grpc_reflection.v1alpha import reflection

            service_names = [
                RobosuitePolicy_pb2.DESCRIPTOR.services_by_name["RobosuitePolicyStepService"].full_name,
                RobosuitePolicy_pb2.DESCRIPTOR.services_by_name["RobosuitePolicyResetService"].full_name,
                RobosuitePolicy_pb2.DESCRIPTOR.services_by_name["RobosuiteGetPolicyMetadataService"].full_name,
                reflection.SERVICE_NAME,
            ]
            if self.config.enable_health_check:
                service_names.append(RobosuitePolicy_pb2.DESCRIPTOR.services_by_name["Health"].full_name)

            reflection.enable_server_reflection(service_names, self.server)

        # Bind server to address
        listen_addr = f"{self.config.host}:{self.config.port}"
        self.server.add_insecure_port(listen_addr)

        # Start batch processing task
        self.batch_processing_task = asyncio.create_task(self._batch_processing_loop())

        # Start server
        await self.server.start()
        log.info(f"RoboVerse gRPC policy server started on {listen_addr}")

        # Set up signal handlers for graceful shutdown
        def signal_handler():
            log.info("Received shutdown signal")
            asyncio.create_task(self.stop())

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    async def stop(self):
        """Stop the gRPC server gracefully."""
        log.info("Stopping RoboVerse gRPC policy server...")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel batch processing
        if self.batch_processing_task:
            self.batch_processing_task.cancel()
            try:
                await self.batch_processing_task
            except asyncio.CancelledError:
                pass

        # Stop server
        if self.server:
            await self.server.stop(grace=5.0)

        # Shutdown policy wrapper
        self.policy_wrapper.shutdown()

        log.info("RoboVerse gRPC policy server stopped")

    async def wait_for_termination(self):
        """Wait for server termination."""
        if self.server:
            await self.server.wait_for_termination()

    async def _batch_processing_loop(self):
        """Main loop for processing batched requests."""
        while not self.shutdown_event.is_set():
            try:
                # Wait for timeout or shutdown
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.config.batch_timeout_ms / 1000.0)
                break  # Shutdown event was set
            except asyncio.TimeoutError:
                # Process any pending requests
                if self.pending_requests:
                    await self._process_batch()

    async def _process_batch(self):
        """Process a batch of pending requests."""
        if not self.pending_requests:
            return

        log.debug(f"Processing batch of {len(self.pending_requests)} requests")

        # Extract requests
        batch_requests = dict(self.pending_requests)
        self.pending_requests.clear()

        # Prepare observations
        observations = {}
        for client_id, (env_state, _, _) in batch_requests.items():
            observations[client_id] = env_state

        try:
            # Get actions from policy
            start_time = time.time()
            actions = self.policy_wrapper.step_batch(observations)
            processing_time = time.time() - start_time

            log.debug(f"Batch processing took {processing_time:.3f}s for {len(observations)} requests")

            # Send responses
            for client_id, (_, response_future, env_id) in batch_requests.items():
                if client_id in actions:
                    action = actions[client_id]
                    grpc_action = action_to_grpc_msg(action, env_id)

                    response = RobosuitePolicy_pb2.RobosuitePolicyStepResponse(success=True, action=grpc_action)
                else:
                    response = RobosuitePolicy_pb2.RobosuitePolicyStepResponse(
                        success=False, error_message=f"No action generated for client {client_id}"
                    )

                if not response_future.done():
                    response_future.set_result(response)

        except Exception as e:
            log.error(f"Error in batch processing: {e}")

            # Send error responses
            for client_id, (_, response_future, _) in batch_requests.items():
                if not response_future.done():
                    response = RobosuitePolicy_pb2.RobosuitePolicyStepResponse(
                        success=False, error_message=f"Batch processing error: {e!s}"
                    )
                    response_future.set_result(response)

    async def add_request(self, client_id: UUID, env_state, env_id: int):
        """Add a request to the batch processing queue.

        Args:
            client_id: Client identifier.
            env_state: Environment state observation.
            env_id: Environment identifier.

        Returns:
            Future that will contain the response.
        """
        response_future = asyncio.get_event_loop().create_future()
        self.pending_requests[client_id] = (env_state, response_future, env_id)

        # If batch is full, process immediately
        if len(self.pending_requests) >= self.config.batch_size:
            await self._process_batch()

        return await response_future


class PolicyStepServicer(RobosuitePolicy_pb2_grpc.RobosuitePolicyStepServiceServicer):
    """gRPC servicer for policy step requests."""

    def __init__(self, server: RobosuitePolicyServer):
        self.server = server

    async def PolicyStep(self, request, context):
        """Handle policy step request."""
        try:
            # Parse request
            client_id = grpc_msg_to_uuid(request.client_identifier)
            env_state, env_id = grpc_msg_to_env_state(request.observation)

            log.debug(f"Received policy step request from client {client_id}, env {env_id}")

            # Add to batch processing queue
            response = await self.server.add_request(client_id, env_state, env_id)
            return response

        except Exception as e:
            log.error(f"Error in PolicyStep: {e}")
            return RobosuitePolicy_pb2.RobosuitePolicyStepResponse(success=False, error_message=f"Server error: {e!s}")


class PolicyResetServicer(RobosuitePolicy_pb2_grpc.RobosuitePolicyResetServiceServicer):
    """gRPC servicer for policy reset requests."""

    def __init__(self, server: RobosuitePolicyServer):
        self.server = server

    async def PolicyReset(self, request, context):
        """Handle policy reset request."""
        try:
            client_id = grpc_msg_to_uuid(request.client_identifier)
            seed = request.seed if request.HasField("seed") else None

            log.debug(f"Received policy reset request from client {client_id} with seed {seed}")

            # Reset policy
            self.server.policy_wrapper.reset_batch({client_id: seed})

            return RobosuitePolicy_pb2.RobosuitePolicyResetResponse(success=True)

        except Exception as e:
            log.error(f"Error in PolicyReset: {e}")
            return RobosuitePolicy_pb2.RobosuitePolicyResetResponse(success=False, error_message=f"Server error: {e!s}")


class GetPolicyMetadataServicer(RobosuitePolicy_pb2_grpc.RobosuiteGetPolicyMetadataServiceServicer):
    """gRPC servicer for policy metadata requests."""

    def __init__(self, server: RobosuitePolicyServer):
        self.server = server

    async def GetPolicyMetadata(self, request, context):
        """Handle policy metadata request."""
        try:
            client_id = grpc_msg_to_uuid(request.client_identifier)

            log.debug(f"Received policy metadata request from client {client_id}")

            # Get metadata from policy wrapper
            metadata = self.server.policy_wrapper.get_policy_metadata()
            grpc_metadata = policy_metadata_to_grpc_msg(**metadata)

            return RobosuitePolicy_pb2.GetPolicyMetadataResponse(success=True, policy_metadata=grpc_metadata)

        except Exception as e:
            log.error(f"Error in GetPolicyMetadata: {e}")
            return RobosuitePolicy_pb2.GetPolicyMetadataResponse(success=False, error_message=f"Server error: {e!s}")


class HealthServicer(RobosuitePolicy_pb2_grpc.HealthServicer):
    """gRPC servicer for health checking."""

    async def Check(self, request, context):
        """Handle health check request."""
        return RobosuitePolicy_pb2.HealthCheckResponse(status=RobosuitePolicy_pb2.HealthCheckResponse.SERVING)


async def run_server(policy_wrapper: PolicyWrapper, config: ServerConfig = None):
    """Run the gRPC policy server.

    Args:
        policy_wrapper: Policy wrapper to serve.
        config: Server configuration (uses defaults if None).
    """
    if config is None:
        config = ServerConfig()

    server = RobosuitePolicyServer(policy_wrapper, config)

    try:
        await server.start()
        await server.wait_for_termination()
    except KeyboardInterrupt:
        log.info("Server interrupted by user")
    finally:
        await server.stop()


def main():
    """Main entry point for running the server."""
    import argparse

    parser = argparse.ArgumentParser(description="RoboVerse gRPC Policy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum worker threads")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--batch-timeout-ms", type=int, default=50, help="Batch timeout in milliseconds")

    args = parser.parse_args()

    config = ServerConfig(
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
    )

    # Example: Create a dummy policy wrapper for testing
    # In practice, you would create and configure your specific policy wrapper here

    from .policy_wrappers.base_wrapper import StatelessPolicyWrapper

    class DummyPolicyWrapper(StatelessPolicyWrapper):
        def step_batch(self, observations):
            # Return dummy actions for all observations
            actions = {}
            for client_id, obs in observations.items():
                # Create dummy action based on observation structure
                action = {}
                for robot_name in obs.get("robots", {}):
                    action[robot_name] = {"dof_pos_target": {}, "dof_effort_target": None}
                actions[client_id] = action
            return actions

        def get_policy_metadata(self):
            return {
                "policy_type": "dummy_policy",
                "model_name": "DummyPolicy",
                "version": "1.0.0",
                "config": {},
                "supported_robots": ["any"],
                "supported_tasks": ["any"],
            }

    dummy_wrapper = DummyPolicyWrapper()

    # Run server
    asyncio.run(run_server(dummy_wrapper, config))


if __name__ == "__main__":
    main()
