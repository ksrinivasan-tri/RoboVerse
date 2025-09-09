"""Conversion utilities between RoboVerse types and gRPC messages."""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import torch

from metasim.types import Action, EnvState, ObjectState, RobotState
from roboverse_grpc.proto import RobosuitePolicy_pb2


def tensor_to_grpc_tensor(tensor: torch.Tensor) -> RobosuitePolicy_pb2.Tensor:
    """Convert PyTorch tensor to gRPC Tensor message."""
    if tensor is None:
        return RobosuitePolicy_pb2.Tensor()

    # Convert to numpy and flatten for serialization
    if isinstance(tensor, torch.Tensor):
        np_tensor = tensor.detach().cpu().numpy()
    else:
        np_tensor = np.array(tensor)

    return RobosuitePolicy_pb2.Tensor(
        shape=list(np_tensor.shape), data=np_tensor.flatten().tolist(), dtype=str(np_tensor.dtype).replace("torch.", "")
    )


def grpc_tensor_to_tensor(grpc_tensor: RobosuitePolicy_pb2.Tensor) -> torch.Tensor:
    """Convert gRPC Tensor message to PyTorch tensor."""
    if not grpc_tensor.data:
        return torch.empty(0)

    # Reconstruct tensor from flattened data
    data = np.array(grpc_tensor.data, dtype=grpc_tensor.dtype)
    if grpc_tensor.shape:
        data = data.reshape(grpc_tensor.shape)

    return torch.from_numpy(data)


def dof_dict_to_grpc_map(dof_dict: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Convert DOF dictionary to gRPC map format."""
    if dof_dict is None:
        return {}
    return dict(dof_dict)


def grpc_map_to_dof_dict(grpc_map: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Convert gRPC map to DOF dictionary."""
    if not grpc_map:
        return None
    return dict(grpc_map)


def robot_state_to_grpc_msg(robot_name: str, robot_state: RobotState) -> RobosuitePolicy_pb2.RobotState:
    """Convert RoboVerse RobotState to gRPC RobotState message."""
    return RobosuitePolicy_pb2.RobotState(
        robot_name=robot_name,
        dof_pos=dof_dict_to_grpc_map(robot_state.get("dof_pos")),
        dof_vel=dof_dict_to_grpc_map(robot_state.get("dof_vel")),
        dof_torque=dof_dict_to_grpc_map(robot_state.get("dof_torque")),
        dof_pos_target=dof_dict_to_grpc_map(robot_state.get("dof_pos_target")),
        dof_vel_target=dof_dict_to_grpc_map(robot_state.get("dof_vel_target")),
        pos=tensor_to_grpc_tensor(robot_state.get("pos")),
        rot=tensor_to_grpc_tensor(robot_state.get("rot")),
        vel=tensor_to_grpc_tensor(robot_state.get("vel")),
        ang_vel=tensor_to_grpc_tensor(robot_state.get("ang_vel")),
        com=tensor_to_grpc_tensor(robot_state.get("com")),
        com_vel=tensor_to_grpc_tensor(robot_state.get("com_vel")),
    )


def grpc_msg_to_robot_state(grpc_robot: RobosuitePolicy_pb2.RobotState) -> tuple[str, RobotState]:
    """Convert gRPC RobotState message to RoboVerse RobotState."""
    robot_state: RobotState = {
        "pos": grpc_tensor_to_tensor(grpc_robot.pos),
        "rot": grpc_tensor_to_tensor(grpc_robot.rot),
        "vel": grpc_tensor_to_tensor(grpc_robot.vel),
        "ang_vel": grpc_tensor_to_tensor(grpc_robot.ang_vel),
        "dof_pos": grpc_map_to_dof_dict(grpc_robot.dof_pos),
        "dof_vel": grpc_map_to_dof_dict(grpc_robot.dof_vel),
        "com": grpc_tensor_to_tensor(grpc_robot.com),
        "com_vel": grpc_tensor_to_tensor(grpc_robot.com_vel),
        "dof_pos_target": grpc_map_to_dof_dict(grpc_robot.dof_pos_target),
        "dof_vel_target": grpc_map_to_dof_dict(grpc_robot.dof_vel_target),
        "dof_torque": grpc_map_to_dof_dict(grpc_robot.dof_torque),
    }
    return grpc_robot.robot_name, robot_state


def object_state_to_grpc_msg(object_name: str, object_state: ObjectState) -> RobosuitePolicy_pb2.ObjectState:
    """Convert RoboVerse ObjectState to gRPC ObjectState message."""
    return RobosuitePolicy_pb2.ObjectState(
        object_name=object_name,
        pos=tensor_to_grpc_tensor(object_state.get("pos")),
        rot=tensor_to_grpc_tensor(object_state.get("rot")),
        vel=tensor_to_grpc_tensor(object_state.get("vel")),
        ang_vel=tensor_to_grpc_tensor(object_state.get("ang_vel")),
        dof_pos=dof_dict_to_grpc_map(object_state.get("dof_pos")),
        dof_vel=dof_dict_to_grpc_map(object_state.get("dof_vel")),
        com=tensor_to_grpc_tensor(object_state.get("com")),
        com_vel=tensor_to_grpc_tensor(object_state.get("com_vel")),
    )


def grpc_msg_to_object_state(grpc_object: RobosuitePolicy_pb2.ObjectState) -> tuple[str, ObjectState]:
    """Convert gRPC ObjectState message to RoboVerse ObjectState."""
    object_state: ObjectState = {
        "pos": grpc_tensor_to_tensor(grpc_object.pos),
        "rot": grpc_tensor_to_tensor(grpc_object.rot),
        "vel": grpc_tensor_to_tensor(grpc_object.vel),
        "ang_vel": grpc_tensor_to_tensor(grpc_object.ang_vel),
        "dof_pos": grpc_map_to_dof_dict(grpc_object.dof_pos),
        "dof_vel": grpc_map_to_dof_dict(grpc_object.dof_vel),
        "com": grpc_tensor_to_tensor(grpc_object.com),
        "com_vel": grpc_tensor_to_tensor(grpc_object.com_vel),
    }
    return grpc_object.object_name, object_state


def camera_data_to_grpc_msg(camera_name: str, camera_data: Dict[str, torch.Tensor]) -> RobosuitePolicy_pb2.CameraData:
    """Convert camera data dictionary to gRPC CameraData message."""
    return RobosuitePolicy_pb2.CameraData(
        camera_name=camera_name,
        rgb=tensor_to_grpc_tensor(camera_data.get("rgb")),
        depth=tensor_to_grpc_tensor(camera_data.get("depth")),
        segmentation=tensor_to_grpc_tensor(camera_data.get("segmentation")),
        intrinsics=tensor_to_grpc_tensor(camera_data.get("intrinsics")),
        pose=tensor_to_grpc_tensor(camera_data.get("pose")),
    )


def grpc_msg_to_camera_data(grpc_camera: RobosuitePolicy_pb2.CameraData) -> tuple[str, Dict[str, torch.Tensor]]:
    """Convert gRPC CameraData message to camera data dictionary."""
    camera_data = {
        "rgb": grpc_tensor_to_tensor(grpc_camera.rgb),
        "depth": grpc_tensor_to_tensor(grpc_camera.depth),
        "segmentation": grpc_tensor_to_tensor(grpc_camera.segmentation),
        "intrinsics": grpc_tensor_to_tensor(grpc_camera.intrinsics),
        "pose": grpc_tensor_to_tensor(grpc_camera.pose),
    }
    return grpc_camera.camera_name, camera_data


def env_state_to_grpc_msg(env_state: EnvState, env_id: int = 0) -> RobosuitePolicy_pb2.RobosuiteObservation:
    """Convert RoboVerse EnvState to gRPC RobosuiteObservation message."""
    # Convert robots
    robot_msgs = []
    for robot_name, robot_state in env_state.get("robots", {}).items():
        robot_msgs.append(robot_state_to_grpc_msg(robot_name, robot_state))

    # Convert objects
    object_msgs = []
    for object_name, object_state in env_state.get("objects", {}).items():
        object_msgs.append(object_state_to_grpc_msg(object_name, object_state))

    # Convert cameras
    camera_msgs = []
    for camera_name, camera_data in env_state.get("cameras", {}).items():
        camera_msgs.append(camera_data_to_grpc_msg(camera_name, camera_data))

    # Create header with current timestamp
    header = RobosuitePolicy_pb2.Header(
        stamp=RobosuitePolicy_pb2.Time(sec=int(time.time()), nanosec=int((time.time() % 1) * 1e9))
    )

    return RobosuitePolicy_pb2.RobosuiteObservation(
        robots=robot_msgs, objects=object_msgs, cameras=camera_msgs, header=header, env_id=env_id
    )


def grpc_msg_to_env_state(grpc_obs: RobosuitePolicy_pb2.RobosuiteObservation) -> tuple[EnvState, int]:
    """Convert gRPC RobosuiteObservation message to RoboVerse EnvState."""
    # Convert robots
    robots = {}
    for robot_msg in grpc_obs.robots:
        robot_name, robot_state = grpc_msg_to_robot_state(robot_msg)
        robots[robot_name] = robot_state

    # Convert objects
    objects = {}
    for object_msg in grpc_obs.objects:
        object_name, object_state = grpc_msg_to_object_state(object_msg)
        objects[object_name] = object_state

    # Convert cameras
    cameras = {}
    for camera_msg in grpc_obs.cameras:
        camera_name, camera_data = grpc_msg_to_camera_data(camera_msg)
        cameras[camera_name] = camera_data

    env_state: EnvState = {"robots": robots, "objects": objects, "cameras": cameras}

    return env_state, grpc_obs.env_id


def action_to_grpc_msg(action: Action, env_id: int = 0) -> RobosuitePolicy_pb2.RobosuiteAction:
    """Convert RoboVerse Action to gRPC RobosuiteAction message."""
    robot_actions = []

    for robot_name, robot_action in action.items():
        robot_action_msg = RobosuitePolicy_pb2.RobotAction(
            robot_name=robot_name,
            dof_pos_target=dof_dict_to_grpc_map(robot_action.get("dof_pos_target")),
            dof_effort_target=dof_dict_to_grpc_map(robot_action.get("dof_effort_target")),
        )
        robot_actions.append(robot_action_msg)

    # Create header with current timestamp
    header = RobosuitePolicy_pb2.Header(
        stamp=RobosuitePolicy_pb2.Time(sec=int(time.time()), nanosec=int((time.time() % 1) * 1e9))
    )

    return RobosuitePolicy_pb2.RobosuiteAction(robot_actions=robot_actions, header=header, env_id=env_id)


def grpc_msg_to_action(grpc_action: RobosuitePolicy_pb2.RobosuiteAction) -> tuple[Action, int]:
    """Convert gRPC RobosuiteAction message to RoboVerse Action."""
    action: Action = {}

    for robot_action_msg in grpc_action.robot_actions:
        robot_action = {
            "dof_pos_target": grpc_map_to_dof_dict(robot_action_msg.dof_pos_target),
            "dof_effort_target": grpc_map_to_dof_dict(robot_action_msg.dof_effort_target),
        }
        action[robot_action_msg.robot_name] = robot_action

    return action, grpc_action.env_id


def uuid_to_grpc_msg(client_uuid: uuid.UUID) -> str:
    """Convert UUID to gRPC client identifier string."""
    return str(client_uuid)


def grpc_msg_to_uuid(client_identifier: str) -> uuid.UUID:
    """Convert gRPC client identifier string to UUID."""
    return uuid.UUID(client_identifier)


def policy_metadata_to_grpc_msg(
    policy_type: str,
    model_name: str,
    version: str,
    config: Dict[str, str] = None,
    supported_robots: List[str] = None,
    supported_tasks: List[str] = None,
) -> RobosuitePolicy_pb2.PolicyMetadata:
    """Create gRPC PolicyMetadata message."""
    return RobosuitePolicy_pb2.PolicyMetadata(
        policy_type=policy_type,
        model_name=model_name,
        version=version,
        config=config or {},
        supported_robots=supported_robots or [],
        supported_tasks=supported_tasks or [],
    )


def grpc_msg_to_policy_metadata(grpc_metadata: RobosuitePolicy_pb2.PolicyMetadata) -> Dict[str, any]:
    """Convert gRPC PolicyMetadata message to dictionary."""
    return {
        "policy_type": grpc_metadata.policy_type,
        "model_name": grpc_metadata.model_name,
        "version": grpc_metadata.version,
        "config": dict(grpc_metadata.config),
        "supported_robots": list(grpc_metadata.supported_robots),
        "supported_tasks": list(grpc_metadata.supported_tasks),
    }
