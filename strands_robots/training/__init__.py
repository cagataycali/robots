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
from strands_robots.training.reward import (
    compute_rabc_weights,
    load_reward_model,
    reward_progress,
)

__all__ = [
    "Trainer",
    "TrainSpec",
    "TrainResult",
    "create_trainer",
    "register_trainer",
    "list_trainers",
    "import_trainer_class",
    "compute_rabc_weights",
    "load_reward_model",
    "reward_progress",
]
