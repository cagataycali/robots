"""The reproducibility seed a TrainSpec asks for is one shared non-negative-count domain.

:attr:`~strands_robots.training.base.TrainSpec.seed` is the field a caller sets
to make a run reproducible, and five backends read it - LeRobot (``cfg.seed`` and
a ``--seed=`` argv token), Cosmos (a ``trainer.seed=`` Hydra override), the two
RL trainers (``torch.manual_seed``) and SageMaker (a string hyperparameter the
job's container seeds from). Before this contract none of them checked it,
and the appliers do not agree about a single value:

* ``torch.manual_seed`` reduces the value modulo ``2**64``, so a negative seed is
  *silently a different seed*: ``manual_seed(-1)`` and ``manual_seed(2**64 - 1)``
  draw the identical stream. Two seeds a caller means to be distinct collapse
  onto one, and the run is reproducible under a number nobody asked for. ``True``
  is a silent seed of ``1`` and ``2.7`` a silent seed of ``2``.
* LeRobot's ``set_seed`` reseeds Python's ``random`` and only *then* hands the
  value to NumPy, which refuses a negative or a float. A refused seed therefore
  leaves the process RNG reseeded by a call that failed, and the message that
  surfaces is NumPy's rather than one naming the field.
* Every value renders into a Hydra override or an argv token, so on those paths a
  bad seed fails - if at all - inside the run, after the dataset and model are
  loaded.

Which backends refuse which values, in which words, is the table in
``tests/training/test_train_spec_field_domains.py``. The tests below ground the
refused values in what the real appliers do with them.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from strands_robots.training.base import TrainSpec

# The backends that seed from the field. The RL trainers are exercised through
# their own spec type further down (their validate() needs an RLTrainSpec).

# Values no applier can honor, split by how each one failed before the gate.

# Silently rewritten by torch's modulo, and refused by NumPy - so no applier
# honors the value the caller supplied.

# Raised out of the applier: torch refuses nan/inf/list, NumPy refuses the float
# and the string.


# Seeds every applier honors as themselves. ``0`` is a seed, not a degenerate
# value, which is why the domain's floor is zero rather than one.


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose ``seed`` is the only thing under test.

    ``validate`` may well report unrelated problems (Cosmos wants a recipe TOML,
    LeRobot a dataset); every assertion below filters for the field name, so an
    unrelated problem can neither mask nor fake a seed verdict.
    """
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="lerobot/act",
        embodiment="new_embodiment",
    )


# The shape the shared domain emits: ``"{context}: seed must be ..."``. Matched
# rather than the bare word because pytest derives ``tmp_path`` from the test
# name, so a path in an unrelated problem can contain "seed" too - a filter that
# picked that up could both mask and fake a verdict.


class TestTheRefusedValuesAreOnesNoApplierCanHonor:
    """Ground the domain in what the real appliers do with each value."""

    @pytest.mark.parametrize(("supplied", "actual"), ((-1, 2**64 - 1), (-5, 2**64 - 5), (True, 1), (2.7, 2)))
    def test_torch_silently_seeds_a_different_stream(self, supplied: Any, actual: int) -> None:
        """The headline: a refused value is not merely rejected downstream.

        ``torch.manual_seed`` reduces its argument, so the run IS reproducible -
        under a seed the caller did not name, and one another caller could have.
        """
        torch = pytest.importorskip("torch")
        torch.manual_seed(supplied)
        as_supplied = torch.rand(8).tolist()
        torch.manual_seed(actual)
        as_rewritten = torch.rand(8).tolist()
        assert as_supplied == as_rewritten, f"seed={supplied!r} no longer collapses onto {actual}"

    def test_numpys_legacy_seeder_refuses_it(self) -> None:
        """The narrowest applier on the LeRobot path refuses what torch rewrites."""
        np = pytest.importorskip("numpy")
        for value in (-1, -5, 2.7, 3.0, float("nan"), float("inf"), "42"):
            with pytest.raises((ValueError, TypeError)):
                np.random.seed(value)

    def test_a_refused_seed_still_reseeds_pythons_random_first(self) -> None:
        """Why the check must precede the applier rather than trust it.

        ``set_seed`` reseeds ``random`` before NumPy refuses, so the failed call
        mutates process state a caller never asked it to touch.
        """
        pytest.importorskip("numpy")
        set_seed = pytest.importorskip("lerobot.utils.random_utils").set_seed
        import random

        random.seed(999)
        untouched = [random.random() for _ in range(2)]
        random.seed(999)
        with pytest.raises((ValueError, TypeError)):
            set_seed(-1)
        after_the_failure = [random.random() for _ in range(2)]
        assert untouched != after_the_failure
