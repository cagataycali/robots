"""wbc_latent - a VLA's SONIC motion tokens decoded into Unitree G1 joint targets.

The blog "Bringing Humanoids to LeRobot" fine-tunes pi0.5 to predict 64-D
SONIC latent motion tokens (plus two gripper commands) instead of joint
angles; on the robot NVIDIA's SONIC decoder turns each token and the last ten
frames of proprioception into 29 joint targets that a per-joint PD law tracks.

This package is that decode stage for Strands Robots:

* :class:`SonicDecoder` - the ONNX decoder with its observation assembly and
  action law reproduced in NumPy (``nvidia/GEAR-SONIC`` ``model_decoder.onnx``).
* :class:`WBCLatentPolicy` - a :class:`~strands_robots.policies.base.Policy`
  wrapping the token-emitting VLA and one decoder; runs once per 50 Hz tick.
* :class:`WBCLatentTorqueController` - the MuJoCo shim applying SONIC's
  armature-derived PD gains to all 29 joints.

Requires the ``[wbc]`` extra (``onnxruntime``). No weights are bundled.
"""

from strands_robots.policies.wbc_latent.constants import (
    SONIC_ACTION_SCALE,
    SONIC_DEFAULT_ANGLES,
    SONIC_JOINT_NAMES,
    SONIC_KDS,
    SONIC_KPS,
    STANDING_TOKEN,
    TOKEN_DIM,
)
from strands_robots.policies.wbc_latent.decoder import (
    SONIC_REPO_ID,
    SONIC_VARIANT_FILES,
    SonicDecoder,
    resolve_decoder_path,
    sonic_variant_error,
)

__all__ = [
    "SONIC_ACTION_SCALE",
    "SONIC_DEFAULT_ANGLES",
    "SONIC_JOINT_NAMES",
    "SONIC_KDS",
    "SONIC_KPS",
    "SONIC_REPO_ID",
    "SONIC_VARIANT_FILES",
    "STANDING_TOKEN",
    "TOKEN_DIM",
    "SonicDecoder",
    "resolve_decoder_path",
    "sonic_variant_error",
]
