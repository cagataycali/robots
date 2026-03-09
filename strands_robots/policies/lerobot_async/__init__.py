#!/usr/bin/env python3
"""
LeRobot Async Inference Policy

Wraps LeRobot's gRPC-based async inference server as a strands-robots Policy.
Supports any LeRobot policy type — validated against LeRobot's own registry,
not a hardcoded list.

The server runs policy inference on GPU while the client (this policy) sends
observations and receives action chunks via gRPC streaming.

Usage:
    # Start LeRobot policy server first:
    # python -m lerobot.async_inference.policy_server --host=localhost --port=8080

    policy = create_policy(
        provider="lerobot_async",
        server_address="localhost:8080",
        policy_type="pi0",
        pretrained_name_or_path="lerobot/pi0-so100-wipe",
        actions_per_chunk=10,
    )
"""

import logging
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .. import Policy

logger = logging.getLogger(__name__)


def _validate_policy_type(policy_type: str) -> None:
    """Validate policy type against LeRobot's own factory.

    No hardcoding — uses LeRobot's get_policy_class() which knows
    all registered policies (including third-party plugins).
    Skips validation if LeRobot isn't installed (let the server handle it).
    """
    try:
        from lerobot.policies.factory import get_policy_class

        get_policy_class(policy_type)  # Raises ValueError if invalid
    except (ImportError, RuntimeError):
        logger.debug("LeRobot not installed locally or has import issues, skipping policy type validation")
    except ValueError:
        # Re-raise with a cleaner message
        try:
            # Try to list what IS available for the error message
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class

            known = PreTrainedConfig.get_known_choices()
            available = sorted(known) if known else "check LeRobot docs"
        except Exception:
            available = "check LeRobot docs"
        raise ValueError(f"Unsupported policy type: '{policy_type}'. " f"LeRobot supports: {available}")


class LerobotAsyncPolicy(Policy):
    """Policy that connects to a LeRobot async inference gRPC server.

    This policy sends robot observations to a remote LeRobot PolicyServer
    and receives action chunks. The server handles model loading, preprocessing,
    inference, and postprocessing.

    Supports any policy type that LeRobot supports — validated dynamically
    against LeRobot's own registry.

    Architecture:
        Robot → observations → gRPC → PolicyServer (GPU) → actions → gRPC → Robot
    """

    def __init__(
        self,
        server_address: str = "localhost:8080",
        policy_type: str = "pi0",
        pretrained_name_or_path: str = "",
        actions_per_chunk: int = 10,
        device: str = "cuda",
        fps: int = 30,
        task: str = "",
        lerobot_features: Optional[Dict] = None,
        rename_map: Optional[Dict[str, str]] = None,
        secure: bool = False,
        tls_credentials: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize LeRobot async policy client.

        Args:
            server_address: gRPC server address (host:port)
            policy_type: LeRobot policy type (act, pi0, smolvla, diffusion, etc.)
            pretrained_name_or_path: HuggingFace model ID or local path
            actions_per_chunk: Number of actions per inference chunk
            device: Device for server-side inference (cuda, cpu)
            fps: Target frames per second
            task: Task instruction string
            lerobot_features: LeRobot feature configuration
            rename_map: Observation key renaming map
            secure: Use TLS for gRPC connection (default: False)
            tls_credentials: Custom gRPC TLS credentials. If secure=True and
                this is None, default SSL channel credentials are used.
        """
        # Validate policy_type against LeRobot's own factory
        _validate_policy_type(policy_type)

        self.server_address = server_address
        self.policy_type = policy_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.actions_per_chunk = actions_per_chunk
        self.device = device
        self.fps = fps
        self.task = task
        self.lerobot_features = lerobot_features or {}
        self.rename_map = rename_map or {}
        self.secure = secure
        self.tls_credentials = tls_credentials
        self.robot_state_keys: List[str] = []

        # gRPC state
        self._channel = None
        self._stub = None
        self._connected = False
        self._timestep = 0

        logger.info(f"🤖 LeRobot Async Policy: {policy_type}")
        logger.info(f"📡 Server: {server_address}")
        logger.info(f"🧠 Model: {pretrained_name_or_path}")
        logger.info(f"⚡ Actions/chunk: {actions_per_chunk}")

    @property
    def provider_name(self) -> str:
        return "lerobot_async"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        """Set robot state keys from connected robot."""
        self.robot_state_keys = robot_state_keys
        logger.info(f"🔧 LeRobot async state keys: {self.robot_state_keys}")

    def _ensure_connected(self):
        """Ensure gRPC connection is established."""
        if self._connected:
            return

        try:
            import grpc
            from lerobot.transport import services_pb2, services_pb2_grpc

            self._channel = (
                grpc.secure_channel(
                    self.server_address,
                    self.tls_credentials or grpc.ssl_channel_credentials(),
                )
                if self.secure
                else grpc.insecure_channel(self.server_address)
            )
            self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)

            # Signal ready
            self._stub.Ready(services_pb2.Empty())

            # Send policy instructions
            from lerobot.async_inference.helpers import RemotePolicyConfig

            remote_config = RemotePolicyConfig(
                policy_type=self.policy_type,
                pretrained_name_or_path=self.pretrained_name_or_path,
                lerobot_features=self.lerobot_features,
                actions_per_chunk=self.actions_per_chunk,
                device=self.device,
                rename_map=self.rename_map,
            )

            config_bytes = pickle.dumps(remote_config)
            self._stub.SendPolicyInstructions(services_pb2.PolicyInstructions(data=config_bytes))

            self._connected = True
            logger.info(f"✅ Connected to LeRobot server at {self.server_address}")

        except ImportError as e:
            raise ImportError(
                f"LeRobot async inference dependencies not available: {e}. " f"Install: pip install lerobot[async]"
            ) from e
        except Exception as e:
            logger.error(f"❌ Failed to connect to LeRobot server: {e}")
            raise

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from LeRobot async inference server.

        Args:
            observation_dict: Robot observations (cameras + state)
            instruction: Natural language instruction
            **kwargs: Additional parameters

        Returns:
            List of action dictionaries for robot execution
        """
        self._ensure_connected()

        try:
            import grpc  # noqa: F401
            from lerobot.async_inference.helpers import TimedObservation
            from lerobot.transport import services_pb2

            # Build raw observation for LeRobot
            raw_obs = self._build_raw_observation(observation_dict, instruction)

            # Create timed observation
            timed_obs = TimedObservation(
                timestamp=time.time(),
                timestep=self._timestep,
                observation=raw_obs,
                must_go=True,
            )
            self._timestep += 1

            # Send observation via gRPC streaming
            obs_bytes = pickle.dumps(timed_obs)

            # Use chunk sending for large observations
            from lerobot.transport.utils import send_bytes_in_chunks

            chunk_iterator = send_bytes_in_chunks(obs_bytes)
            self._stub.SendObservations(chunk_iterator)

            # Request actions
            actions_response = self._stub.GetActions(services_pb2.Empty())

            if not actions_response.data:
                logger.warning("⚠️ Empty action response from server")
                return self._generate_zero_actions()

            # Deserialize action chunk
            # NOTE: pickle.loads is used here because the server serializes with pickle.
            # This assumes the inference server is trusted (e.g. running locally or on
            # a private network). Do NOT connect to untrusted servers — pickle
            # deserialization of untrusted data can lead to arbitrary code execution.
            # TODO: Consider migrating to a safer serialization format (protobuf/msgpack).
            timed_actions = pickle.loads(actions_response.data)  # noqa: S301

            if not isinstance(timed_actions, list):
                logger.warning("Unexpected action type from server, expected list")
                return self._generate_zero_actions()

            # Convert TimedAction list to robot action dicts
            return self._convert_actions(timed_actions)

        except Exception as e:
            logger.error(f"❌ LeRobot async inference error: {e}")
            return self._generate_zero_actions()

    def _build_raw_observation(self, observation_dict: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """Convert strands-robots observation to LeRobot raw observation format."""
        raw_obs = {}

        # Add state values
        for key in self.robot_state_keys:
            if key in observation_dict:
                raw_obs[key] = observation_dict[key]

        # Add camera images
        for key, value in observation_dict.items():
            if key not in self.robot_state_keys:
                if isinstance(value, np.ndarray) and value.ndim >= 2:
                    raw_obs[key] = value

        # Add task instruction
        if instruction:
            raw_obs["task"] = instruction

        return raw_obs

    def _convert_actions(self, timed_actions) -> List[Dict[str, Any]]:
        """Convert LeRobot TimedAction list to robot action dictionaries."""
        robot_actions = []

        for timed_action in timed_actions:
            action_tensor = timed_action.get_action()
            action_array = action_tensor.numpy() if hasattr(action_tensor, "numpy") else np.array(action_tensor)

            action_dict = {}
            for j, key in enumerate(self.robot_state_keys):
                if j < len(action_array):
                    action_dict[key] = float(action_array[j])
                else:
                    action_dict[key] = 0.0
            robot_actions.append(action_dict)

        logger.debug(f"⚡ Converted {len(robot_actions)} actions from LeRobot server")
        return robot_actions

    def _generate_zero_actions(self) -> List[Dict[str, Any]]:
        """Generate zero actions as fallback."""
        return [{key: 0.0 for key in self.robot_state_keys} for _ in range(self.actions_per_chunk)]

    def disconnect(self):
        """Disconnect from gRPC server."""
        if self._channel:
            self._channel.close()
            self._connected = False
            logger.info("🔌 Disconnected from LeRobot server")

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass


__all__ = ["LerobotAsyncPolicy"]
