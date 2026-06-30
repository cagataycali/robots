"""From-scratch reinforcement-learning trainers for ``strands_robots``.

The RL peer of the supervised ``Trainer`` family: train a policy *from a reward
function* by interacting with a :class:`~strands_robots.training.rl.env.SimEnv`
(a Gym-style wrapper over a ``SimEngine``), rather than post-tuning from a
dataset. Selected through the same ``create_trainer`` factory
(``create_trainer("ppo")``).

Public surface:
    - :class:`BaseRLAlgo` - abstract RL trainer (peer of ``Trainer``).
    - :class:`RLTrainSpec` - reward-driven training spec.
    - :class:`PpoTrainer` - Proximal Policy Optimization (on-policy).
    - :class:`FastSacTrainer` - Soft Actor-Critic (off-policy, replay buffer).
    - :class:`SimpleReplayBuffer` - off-policy transition store.
    - :class:`SimEnv` - ``SimEngine`` -> RL env adapter.
    - :class:`VecSimEnv` - N independent ``SimEnv`` presented as one ``(N, D)`` env.
    - :class:`EmpiricalNormalization` - running observation normalizer.

Importing this package imports ``torch`` (via the env / algo modules), so it is
not imported by ``strands_robots.training.__init__``; the ``ppo`` provider is
registered there through a lazy loader instead.
"""

from strands_robots.training.rl.base_algo import BaseRLAlgo, RLTrainSpec
from strands_robots.training.rl.env import SimEnv
from strands_robots.training.rl.fast_sac import FastSacTrainer
from strands_robots.training.rl.gym_env import GymSimEnv
from strands_robots.training.rl.normalization import EmpiricalNormalization
from strands_robots.training.rl.ppo import PpoTrainer
from strands_robots.training.rl.replay_buffer import SimpleReplayBuffer
from strands_robots.training.rl.vec_env import VecSimEnv

__all__ = [
    "BaseRLAlgo",
    "RLTrainSpec",
    "PpoTrainer",
    "FastSacTrainer",
    "SimpleReplayBuffer",
    "SimEnv",
    "GymSimEnv",
    "VecSimEnv",
    "EmpiricalNormalization",
]
