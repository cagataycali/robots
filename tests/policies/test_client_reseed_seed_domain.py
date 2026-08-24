"""The client-side reseed accepts exactly the seeds it can apply to every RNG.

:func:`~strands_robots.policies._rng.reseed_client_rngs` is the applier both
service-mode providers route ``Policy.reset`` through, so it is how a rollout
makes an episode reproducible. It reseeds Python ``random``, then NumPy's
legacy global RNG, then torch - three appliers with three different accepted
domains, run in sequence inside one swallowing ``try``.

Only the second of them bounds the value, so before the guard under test every
seed NumPy refuses left the process *half* seeded: Python ``random`` reseeded,
NumPy untouched, ``reset`` returning normally, and the reason only at ``INFO``.
A caller was told the episode was seeded while half the streams a stochastic
policy draws from were not - the failure mode that is worse than a refusal,
because nothing distinguishes it from success.

What is pinned here:

* every seed the helper cannot apply is refused *before the first applier
  runs*, so neither RNG moves;
* every seed it accepts reaches all three appliers, so an accepted value can
  never leave a partial state either;
* the accepted domain is the one the sibling applier
  :func:`~strands_robots.simulation.policy_runner.set_eval_seed` already
  enforces - both reseed the same legacy NumPy global RNG, so they can honor
  the same seeds and a divergence between them is a defect in whichever moved;
* the best-effort swallow the module documents still covers what it was
  written for - an applier that *fails* - which is a different case from a
  value no applier can accept.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
import pathlib
import random
import sys
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from strands_robots.policies import _rng
from strands_robots.policies._rng import reseed_client_rngs
from strands_robots.simulation.base import MAX_EVAL_SEED
from strands_robots.simulation.policy_runner import set_eval_seed

#: Seeds this path cannot apply. Each is refused by NumPy's legacy global RNG,
#: which is the narrowest applier: negatives and anything above
#: :data:`MAX_EVAL_SEED` are out of its range, a fractional *or integral* float
#: is refused by its dtype cast, and ``bool`` is an ``int`` subclass whose
#: ``True`` would install a silent seed of 1. Python ``random`` accepts most of
#: them, which is exactly why they used to half-seed the process.
UNUSABLE_SEEDS: list[Any] = [
    -1,
    -5,
    2.5,
    3.0,
    True,
    False,
    "42",
    [7],
    math.nan,
    math.inf,
    MAX_EVAL_SEED + 1,
    2**64,
]

#: Seeds every applier on this path honors, including both endpoints of the
#: accepted range - ``0`` is a seed, not the absence of one.
USABLE_SEEDS: list[Any] = [0, 1, 7, 12345, MAX_EVAL_SEED]

#: Baseline the RNGs are put into before each measurement. Kept out of both
#: lists above so "the state changed" is never true by coincidence.
BASELINE_SEED = 999


@pytest.fixture(autouse=True)
def _restore_global_rngs() -> Iterator[None]:
    """Restore the process-wide RNGs this module reseeds.

    The subject is a helper whose whole job is a global side effect, so every
    test here mutates interpreter-wide state. Restoring it keeps that from
    reaching an unrelated test through execution order.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def _rng_fingerprint() -> tuple[Any, Any]:
    """A comparable snapshot of the two global RNGs the helper reseeds.

    ``get_state`` is annotated as returning the keyword-argument form, so the
    legacy tuple it actually returns here is read through an ``Any`` binding.
    """
    numpy_state: Any = np.random.get_state()
    return random.getstate(), (tuple(int(v) for v in numpy_state[1][:8]), int(numpy_state[2]))


def _at_baseline() -> tuple[Any, Any]:
    """Put both RNGs in a known state and return its fingerprint."""
    random.seed(BASELINE_SEED)
    np.random.seed(BASELINE_SEED)
    return _rng_fingerprint()


def _verdict(applier: Callable[[Any], None], seed: Any) -> str:
    """Classify what *applier* does with *seed*.

    The seeds below are deliberately outside the ``int`` the appliers annotate,
    so every call goes through this one funnel: mypy does not narrow an ``Any``
    argument, and a single place to widen keeps the parity comparison honest.
    Only ``ValueError`` counts as a refusal - anything else escaping is a
    finding rather than a verdict.
    """
    _at_baseline()
    try:
        applier(seed)
    except ValueError:
        return "refused"
    return "applied"


class TestAnUnusableSeedIsRefusedBeforeAnyRngIsTouched:
    """The guard's placement is the fix: a refusal must move no RNG at all."""

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS, ids=[repr(s) for s in UNUSABLE_SEEDS])
    def test_it_raises_and_leaves_both_rngs_untouched(self, seed: Any) -> None:
        before = _at_baseline()
        with pytest.raises(ValueError, match=r"reseed_client_rngs: seed must be"):
            reseed_client_rngs(seed)
        assert _rng_fingerprint() == before, (
            f"a refused seed ({seed!r}) reseeded one of the global RNGs; the guard has to run "
            "before the first applier or the process is left half seeded"
        )

    def test_the_reason_names_the_helper_and_the_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            reseed_client_rngs(-1)
        text = str(excinfo.value)
        assert text.startswith("reseed_client_rngs: "), text
        assert "-1" in text, text

    def test_the_ceiling_reason_names_the_bound_that_produced_it(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            reseed_client_rngs(MAX_EVAL_SEED + 1)
        assert f"[0, {MAX_EVAL_SEED}]" in str(excinfo.value), str(excinfo.value)

    def test_none_is_not_offered_as_a_remedy(self) -> None:
        """``None`` returned earlier as a documented no-op, so a value that
        reached the guard cannot be fixed by passing it."""
        # Bound through ``Any`` because the point is what the runtime does with
        # a value outside the annotated ``int | None``.
        fractional: Any = 2.5
        with pytest.raises(ValueError) as excinfo:
            reseed_client_rngs(fractional)
        assert "or None" not in str(excinfo.value), str(excinfo.value)


class TestEverySeedItAcceptsReachesEveryApplier:
    """The other half of all-or-nothing, and what stops the guard over-reaching."""

    @pytest.mark.parametrize("seed", USABLE_SEEDS, ids=[repr(s) for s in USABLE_SEEDS])
    def test_python_and_numpy_are_both_reseeded(self, seed: Any) -> None:
        before = _at_baseline()
        reseed_client_rngs(seed)
        after = _rng_fingerprint()
        assert after[0] != before[0], f"Python random was not reseeded by {seed!r}"
        assert after[1] != before[1], f"the NumPy global RNG was not reseeded by {seed!r}"

    @pytest.mark.parametrize("seed", USABLE_SEEDS, ids=[repr(s) for s in USABLE_SEEDS])
    def test_torch_is_reseeded_too(self, seed: Any) -> None:
        torch = pytest.importorskip("torch")
        torch.manual_seed(BASELINE_SEED)
        reseed_client_rngs(seed)
        assert torch.initial_seed() == int(seed), (
            f"torch kept its previous seed after reseed_client_rngs({seed!r}); every accepted "
            "value has to reach every applier or an accepted seed can half-seed the process too"
        )

    def test_an_accepted_seed_is_still_reproducible(self) -> None:
        reseed_client_rngs(4242)
        first = ([random.random() for _ in range(3)], np.random.rand(3).tolist())
        reseed_client_rngs(4242)
        second = ([random.random() for _ in range(3)], np.random.rand(3).tolist())
        assert first == second


class TestTheDomainMatchesTheSiblingApplier:
    """Both appliers reseed the same legacy NumPy global RNG, so both can honor
    exactly the same seeds. Pinned as an equality rather than assumed from the
    shared helper, so either side drifting fails here."""

    @pytest.mark.parametrize(
        "seed", UNUSABLE_SEEDS + USABLE_SEEDS, ids=[repr(s) for s in UNUSABLE_SEEDS + USABLE_SEEDS]
    )
    def test_the_two_appliers_agree(self, seed: Any) -> None:
        assert _verdict(reseed_client_rngs, seed) == _verdict(set_eval_seed, seed), (
            f"reseed_client_rngs and set_eval_seed disagree about {seed!r}; they apply the same "
            "seed to the same RNGs, so a value one refuses the other cannot honor"
        )

    def test_the_probe_set_exercises_both_verdicts(self) -> None:
        """Non-vacuity: the parity assertion above is worthless if every row
        lands on one verdict."""
        verdicts = {_verdict(set_eval_seed, seed) for seed in UNUSABLE_SEEDS + USABLE_SEEDS}
        assert verdicts == {"refused", "applied"}, verdicts


class TestTheBestEffortSwallowIsPreserved:
    """The guard narrows which values are accepted. It does not change what
    happens when an applier itself fails, which is the case the module's
    best-effort clause was written for."""

    def test_a_failing_applier_is_still_logged_and_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def boom(_seed: Any) -> None:
            raise RuntimeError("simulated RNG backend failure")

        monkeypatch.setattr(np.random, "seed", boom)
        with caplog.at_level(logging.INFO, logger=_rng.logger.name):
            reseed_client_rngs(7)  # must not raise
        assert any("reseed failed" in record.getMessage() for record in caplog.records), [
            record.getMessage() for record in caplog.records
        ]

    def test_a_missing_torch_is_still_skipped_silently(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        with caplog.at_level(logging.INFO, logger=_rng.logger.name):
            reseed_client_rngs(7)
        assert not [r for r in caplog.records if "reseed failed" in r.getMessage()], [
            r.getMessage() for r in caplog.records
        ]


class TestNoneIsStillANoOp:
    def test_none_touches_neither_rng(self) -> None:
        before = _at_baseline()
        reseed_client_rngs(None)
        assert _rng_fingerprint() == before


def test_the_shared_domain_is_imported_lazily() -> None:
    """Reaching :mod:`strands_robots.simulation.base` from module scope here
    would close an import ring - it imports ``policy_runner``, which imports
    ``policies.base`` - and a cyclic module-level import is reported as
    possibly-undefined rather than merely untidy. Pinned because the deferred
    form looks like something to tidy up."""
    source = pathlib.Path(inspect.getfile(_rng)).read_text(encoding="utf-8")
    leaked = [
        ast.unparse(node)
        for node in ast.parse(source).body
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("strands_robots.simulation")
    ]
    assert not leaked, f"import must stay inside the function: {leaked}"
