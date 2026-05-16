"""Cosmos Predict 2.5 policy provider for strands-robots.

Wraps NVIDIA's Cosmos-Predict2.5/robot/policy checkpoint for direct
action prediction via latent-diffusion denoising. Post-trained on
LIBERO (98.5% success) and RoboCasa benchmarks.

Architecture:
    [Camera Images + Proprio + Language] -> VAE Encoder -> Latent Sequence
    -> Rectified Flow DiT (2B) -> Denoised Latent
    -> Extract Action Chunk (16-step, 7-DoF)

Requirements:
    - cosmos-predict2 package (from nvidia-cosmos/cosmos-predict2.5)
    - CUDA GPU with 16GB+ VRAM

Usage::

    from strands_robots.policies import create_policy

    policy = create_policy(
        "cosmos_predict",
        model_id="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        suite="libero",
    )

Reference:
    "Cosmos World Foundation Model Platform for Physical AI", arXiv:2511.00062
    GitHub: https://github.com/nvidia-cosmos/cosmos-predict2.5
"""

from strands_robots.policies.cosmos_predict.policy import CosmosPredictPolicy

__all__ = ["CosmosPredictPolicy"]
