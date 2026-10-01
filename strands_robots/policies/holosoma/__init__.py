"""Holosoma policy - Amazon FAR whole-body locomotion for the Unitree G1.

:class:`HolosomaPolicy` runs the released ``holosoma_inference`` locomotion
checkpoints (Apache-2.0, https://github.com/amazon-far/holosoma) in process with
ONNX Runtime, the second whole-body-control family next to the GR00T-WBC ports in
:mod:`strands_robots.policies.wbc`. Both drive the same 29-joint table, both emit
joint-position targets, and on MuJoCo both run through the same PD-to-torque shim.

Requires the ``[holosoma]`` extra (``onnxruntime`` + ``huggingface_hub``); no
weights are bundled.
"""

from strands_robots.policies.holosoma.config import (
    HOLOSOMA_G1_DEFAULT_ANGLES,
    HOLOSOMA_OBS_DIM,
    HolosomaConfig,
)
from strands_robots.policies.holosoma.observation import (
    ACTOR_OBS_DIMS,
    ACTOR_OBS_TERMS,
    GaitPhase,
    actor_obs_slices,
    build_actor_obs,
)
from strands_robots.policies.holosoma.policy import (
    HOLOSOMA_FILES,
    HOLOSOMA_G1_JOINTS,
    HOLOSOMA_HF_REPO,
    HOLOSOMA_HF_REVISION,
    HOLOSOMA_SHA256,
    HolosomaPolicy,
    read_onnx_metadata,
    resolve_holosoma_checkpoint,
)

__all__ = [
    "ACTOR_OBS_DIMS",
    "ACTOR_OBS_TERMS",
    "HOLOSOMA_FILES",
    "HOLOSOMA_G1_DEFAULT_ANGLES",
    "HOLOSOMA_G1_JOINTS",
    "HOLOSOMA_HF_REPO",
    "HOLOSOMA_HF_REVISION",
    "HOLOSOMA_SHA256",
    "HOLOSOMA_OBS_DIM",
    "GaitPhase",
    "HolosomaConfig",
    "HolosomaPolicy",
    "actor_obs_slices",
    "build_actor_obs",
    "read_onnx_metadata",
    "resolve_holosoma_checkpoint",
]
