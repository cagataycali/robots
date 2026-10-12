"""The launch topology a TrainSpec asks for is one shared positive-count domain.

:attr:`~strands_robots.training.base.TrainSpec.num_gpus` and
:attr:`~strands_robots.training.base.TrainSpec.num_nodes` are the two process
counts a distributed run is sized from. Each is read in three places by the
three supervised backends: a ``spec.num_gpus > 1`` / ``spec.num_nodes > 1`` test
that selects between the single-process and the multi-process launch path, a
``nproc_per_node`` / ``nnodes`` argument to torch's ``elastic_launch``, and a
``--nproc_per_node=`` / ``--nnodes=`` / ``--num_gpus=`` argv token.

Before this contract neither field had a domain, while their run-size
neighbours ``steps`` and ``global_batch_size`` shared one - and every way a bad
process count failed was silent or late:

* ``0``, a negative, ``nan`` and ``True`` all compare as *not* greater than one
  (``nan`` compares false against everything), so the selector routed them to
  the single-process path and the run proceeded on one process under a
  successful result. For ``num_nodes`` that also slipped past the multi-node
  refusal LeRobot and GR00T raise for a topology they cannot run.
* ``2.7`` and ``inf`` *are* greater than one, so they selected the multi-process
  path and reached ``elastic_launch`` as the worker count.
  ``torch.distributed.launcher.api.LaunchConfig`` accepts both without
  complaint, so nothing downstream rejected them either.
* A string, ``None`` or a list raised ``TypeError`` out of the comparison
  itself - from inside a :meth:`Trainer.validate` documented to *return*
  problems.

Which backends refuse which values, in which words, is the table in
``tests/training/test_train_spec_field_domains.py``. The tests below pin that
the multi-node refusal still fires for a usable count.
"""

from __future__ import annotations

import pathlib

import pytest

from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.lerobot import LerobotTrainer


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose topology fields are the only thing under test.

    ``validate`` may well report unrelated problems (GR00T wants a checkout,
    Cosmos a recipe TOML); every assertion below filters for the field name, so
    an unrelated problem cannot mask - or fake - a topology verdict.
    """
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="lerobot/act",
        embodiment="new_embodiment",
    )


def _problems_about(trainer: Trainer, spec: TrainSpec, field: str) -> list[str]:
    """``validate`` problems that name *field*."""
    return [p for p in trainer.validate(spec) if field in p]


class TestAUsableTopologyIsUntouched:
    """A usable process count raises no topology problem on any backend."""

    @pytest.mark.parametrize("trainer_cls", (LerobotTrainer,))
    def test_the_multi_node_refusal_still_fires_for_a_usable_count(
        self, spec: TrainSpec, trainer_cls: type[Trainer]
    ) -> None:
        """The guarded comparison must still run once the count IS a count.

        LeRobot and GR00T cannot launch multi-node in-process and say so. That
        comparison is now reached only when the shared domain has established
        ``num_nodes`` is a count - so this pins that gating it did not disable
        it.
        """
        spec.num_nodes = 8
        problems = _problems_about(trainer_cls(), spec, "num_nodes")
        assert any("multi-node" in p for p in problems), problems
        assert not any("must be a positive integer" in p for p in problems), problems
