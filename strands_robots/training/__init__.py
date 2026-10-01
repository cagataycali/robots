"""Training abstraction - post-tune any policy provider natively.

Peer of ``strands_robots.policies``: where ``Policy`` is inference, ``Trainer``
is post-tuning. Selected by the SAME provider name via ``create_trainer``.

Usage::

    from strands_robots.training import create_trainer, TrainSpec

    trainer = create_trainer("lerobot_local")
    spec = TrainSpec(
        dataset_root="/tmp/my_dataset",
        base_model="lerobot/act_aloha_sim",
        output_dir="/tmp/ft_out",
        steps=20000,
    )
    problems = trainer.validate(spec)
    if not problems:
        result = trainer.train(spec)
"""

from strands_robots.training.base import Trainer, TrainResult, TrainSpec
from strands_robots.training.factory import (
    create_trainer,
    import_trainer_class,
    list_trainers,
    register_trainer,
)

__all__ = [
    "Trainer",
    "TrainSpec",
    "TrainResult",
    "create_trainer",
    "register_trainer",
    "list_trainers",
    "import_trainer_class",
]


# Register the from-scratch RL trainers (strands_robots.training.rl). These live
# in a torch-importing subpackage, so they are wired through the factory's lazy
# loader rather than imported here - keeping ``import strands_robots.training``
# torch-free. ``create_trainer("ppo")`` resolves the loader on first use.
def _load_ppo_trainer() -> type[Trainer]:
    from strands_robots.training.rl.ppo import PpoTrainer

    return PpoTrainer


register_trainer("ppo", _load_ppo_trainer)


def _load_fast_sac_trainer() -> type[Trainer]:
    from strands_robots.training.rl.fast_sac import FastSacTrainer

    return FastSacTrainer


register_trainer("fast_sac", _load_fast_sac_trainer)


def _load_fast_td3_trainer() -> type[Trainer]:
    from strands_robots.training.rl.fast_td3 import FastTd3Trainer

    return FastTd3Trainer


register_trainer("fast_td3", _load_fast_td3_trainer)


# Register the SageMaker managed-job transport. Auto-discovery would resolve
# ``create_trainer("sagemaker")`` from the module name alone, but registration
# is what puts the provider in ``list_trainers()`` (there is no policy-side
# ``"trainer"`` block to list it - "sagemaker" is a training transport with no
# paired inference provider). The loader keeps the import deferred; the module
# itself defers boto3 to first use via ``require_optional``.
def _load_sagemaker_trainer() -> type[Trainer]:
    from strands_robots.training.sagemaker import SagemakerTrainer

    return SagemakerTrainer


register_trainer("sagemaker", _load_sagemaker_trainer)


# Register the Isaac Lab transport. Like "sagemaker" it has no paired inference
# provider, so registration is what lists it. The trainer imports no Isaac Lab
# code at all - it launches the CLI of a separate interpreter ($ISAACLAB_PYTHON).
def _load_isaaclab_trainer() -> type[Trainer]:
    from strands_robots.training.isaaclab import IsaacLabTrainer

    return IsaacLabTrainer


register_trainer("isaaclab", _load_isaaclab_trainer)


# ``rsl_rl`` is the training-side name of the ``rsl_rl_onnx`` provider (mjlab +
# rsl_rl PPO in, ONNX out). The policies.json ``trainer`` block already resolves
# ``create_trainer("rsl_rl_onnx")``; this alias lets an agent say what it is
# doing (training with rsl_rl) rather than what it will load afterwards.
def _load_rsl_rl_trainer() -> type[Trainer]:
    from strands_robots.training.rsl_rl import RslRlTrainer

    return RslRlTrainer


register_trainer("rsl_rl", _load_rsl_rl_trainer, aliases=["mjlab"])
