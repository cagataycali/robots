"""``set_eval_seed`` requires a seed - ``None`` is refused, not half-applied.

``set_eval_seed`` is public API (exported via ``__all__`` and documented for
standalone callers who drive a rollout without going through
``evaluate_benchmark``). Its domain is
:func:`~strands_robots.simulation.base.randomization_seed_error`, which accepts
``None`` because for ``randomize(seed=None)`` / ``set_obs_noise(seed=None)`` that
is a legitimate parameter value meaning "draw fresh entropy".

For the applier there is no seed to apply, and the three RNGs it drives disagree
about it. ``random.seed(None)`` and ``numpy.random.seed(None)`` reseed from
entropy; ``torch.manual_seed(None)`` raises ``TypeError: int() argument must be a
string, a bytes-like object or a real number, not 'NoneType'``. So the call
reseeded two global RNGs and *then* raised - a partial side effect from a call
that failed - with a message naming neither the parameter nor the method. On an
install without torch the same call raised nothing at all and reseeded both from
entropy, so the two installs disagreed about whether anything had happened.

Either way an unseeded rollout acquires a process-wide RNG side effect it never
asked for, which is the opposite of the rule ``PolicyRunner.evaluate`` states one
layer up: "a ``None`` seed leaves the master RNG unbuilt rather than seeding it
from entropy: an unseeded eval must not acquire a global RNG side effect it never
had."

The domain now carries ``allow_none``, the same per-destination shape as the
existing ``max_seed`` ceiling: ``None`` stays valid where it selects fresh
entropy or means "do not seed", and the applier opts out. These tests pin the
refusal, the absence of any side effect on the refusal path, that both installs
agree on it, and - deliberately - that the shared domain keeps accepting ``None``,
so a later change that "harmonises" the two fails here instead of silently
reintroducing the entropy side effect.

Nothing here needs a simulator or torch: ``set_eval_seed`` is a pure RNG helper
and both heavy imports inside it are lazy. That is the point of the
torch-absent case, so this module deliberately does not import the sibling
``test_rollout_seed_is_applied_or_refused``, whose module-level
``importorskip("mujoco")`` would skip it on exactly the minimal install the
contract has to hold on. The facade side of the boundary - an unseeded
``run_policy`` / ``eval_policy`` still succeeding at ``seed=None`` - is pinned
there.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable
from typing import Any

import pytest

from strands_robots.simulation.base import MAX_EVAL_SEED, randomization_seed_error
from strands_robots.simulation.policy_runner import set_eval_seed

# Values no RNG can be seeded from, mirroring the sibling module's list. Kept
# local rather than imported for the reason in the module docstring.
UNUSABLE_SEEDS: list[Any] = [-1, 2.7, 3.0, True, False, math.nan, math.inf, "42", [1]]

# What the applier can honor. The rollout facades' usable set is this plus
# ``None``, and that one difference is the asymmetry this module exists to pin.
APPLIER_USABLE_SEEDS: list[int] = [0, 7, 2**31, MAX_EVAL_SEED]

_PROBE_SEED = 0xC0FFEE

# The bare message the torch branch used to leak, asserted absent below.
_TORCH_TYPE_ERROR_FRAGMENT = "int() argument must be a string"


def _apply(seed: Any) -> None:
    """Call the applier with a value its ``seed: int`` annotation excludes.

    A single funnel so the deliberately off-type arguments below need one
    documented ``Any`` rather than a suppression at every call site.
    """
    set_eval_seed(seed)


def _refusal_moved_python_rng(call: Callable[[], None]) -> bool:
    """Whether a refused call moved Python's global RNG.

    Asserts the refusal itself, so the measurement cannot pass vacuously on a
    build where the call succeeds silently instead of raising.
    """
    random.seed(_PROBE_SEED)
    before = [random.random() for _ in range(3)]
    random.seed(_PROBE_SEED)
    with pytest.raises(ValueError):
        call()
    after = [random.random() for _ in range(3)]
    return before != after


class TestTheApplierRequiresASeed:
    """``None`` is the absence of a seed, so there is nothing to apply."""

    def test_none_raises_a_named_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"set_eval_seed: seed is required"):
            _apply(None)

    def test_the_bare_torch_type_error_no_longer_escapes(self) -> None:
        """The old failure named neither the parameter nor the method."""
        with pytest.raises(ValueError) as excinfo:
            _apply(None)
        assert _TORCH_TYPE_ERROR_FRAGMENT not in str(excinfo.value)
        assert "NoneType" not in str(excinfo.value)

    def test_the_message_names_what_to_do_instead(self) -> None:
        """A refusal with no alternative is a dead end: the caller wanting the
        RNGs untouched has to be told not to call this at all."""
        with pytest.raises(ValueError) as excinfo:
            _apply(None)
        text = str(excinfo.value)
        assert "do not call set_eval_seed" in text
        assert "not a seed to apply" in text


class TestTheRefusalTouchesNoRng:
    """The check sits ahead of every RNG, so the partial reseed goes with it."""

    def test_the_python_random_stream_is_unchanged(self) -> None:
        assert _refusal_moved_python_rng(lambda: _apply(None)) is False

    def test_the_numpy_global_stream_is_unchanged(self) -> None:
        np = pytest.importorskip("numpy")
        np.random.seed(_PROBE_SEED)
        before = np.random.random(3).tolist()
        np.random.seed(_PROBE_SEED)
        with pytest.raises(ValueError):
            _apply(None)
        assert np.random.random(3).tolist() == before


class TestBothInstallsAgreeAboutIt:
    """Without torch the call used to succeed silently and reseed from entropy,
    so the two installs disagreed about whether anything had happened."""

    @pytest.fixture
    def without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A ``None`` entry makes ``import torch`` raise ImportError, which the
        # applier's own handler catches - the pattern the RNG-parity and
        # norm-stats tests already use for a torch-free install.
        monkeypatch.setitem(sys.modules, "torch", None)

    def test_the_same_value_error_is_raised(self, without_torch: None) -> None:
        with pytest.raises(ValueError, match=r"set_eval_seed: seed is required"):
            _apply(None)

    def test_no_rng_is_reseeded_from_entropy(self, without_torch: None) -> None:
        assert _refusal_moved_python_rng(lambda: _apply(None)) is False

    def test_a_usable_seed_is_still_applied(self, without_torch: None) -> None:
        """Over-reach control: the torch-free path still seeds what it can."""
        set_eval_seed(7)
        first = [random.random() for _ in range(3)]
        set_eval_seed(7)
        assert [random.random() for _ in range(3)] == first


class TestTheAsymmetryIsDeliberate:
    """``None`` is a valid *parameter* for the randomize family and not a valid
    *seed* for anything, so the two domains differ on exactly one value."""

    def test_the_shared_domain_still_accepts_none(self) -> None:
        """``randomize(seed=None)`` depends on it, and the message documents it
        as the fresh-entropy spelling."""
        assert randomization_seed_error(None, "randomize") is None
        assert randomization_seed_error(None, "set_obs_noise") is None

    def test_the_rollout_facades_still_accept_none(self) -> None:
        """At a facade ``None`` means "do not seed", which is why every caller
        guards with ``if seed is not None`` rather than forwarding it."""
        assert randomization_seed_error(None, "PolicyRunner.run", max_seed=MAX_EVAL_SEED) is None
        assert randomization_seed_error(None, "run_policy", max_seed=MAX_EVAL_SEED) is None

    def test_the_applier_refuses_the_one_value_the_shared_domain_accepts(self) -> None:
        """Both halves in one assertion: a later change that harmonises them by
        accepting ``None`` at the applier fails here instead of silently
        reintroducing the entropy side effect."""
        assert randomization_seed_error(None, "set_eval_seed", max_seed=MAX_EVAL_SEED) is None
        with pytest.raises(ValueError, match=r"seed is required"):
            _apply(None)

    def test_the_opt_out_is_what_refuses_it(self) -> None:
        assert randomization_seed_error(None, "set_eval_seed", allow_none=False) is not None
        assert randomization_seed_error(None, "set_eval_seed", allow_none=True) is None


class TestTheMessagesDescribeTheDomainTheCallerHas:
    """A reason that offers a value the destination refuses is a dead end."""

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS, ids=repr)
    def test_the_applier_never_advertises_none(self, seed: Any) -> None:
        reason = randomization_seed_error(seed, "set_eval_seed", max_seed=MAX_EVAL_SEED, allow_none=False)
        assert reason is not None
        assert "None" not in reason

    def test_the_ceiling_message_does_not_advertise_none_either(self) -> None:
        reason = randomization_seed_error(MAX_EVAL_SEED + 1, "set_eval_seed", max_seed=MAX_EVAL_SEED, allow_none=False)
        assert reason is not None
        assert "None" not in reason
        assert f"[0, {MAX_EVAL_SEED}]" in reason

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS, ids=repr)
    def test_every_default_path_message_is_unchanged(self, seed: Any) -> None:
        """The opt-out is per destination: every caller that keeps ``None`` gets
        byte-identical text, so this change refuses nothing it did not before."""
        reason = randomization_seed_error(seed, "randomize")
        assert reason is not None
        assert reason == (
            f"randomize: seed must be a non-negative integer or None, got {seed!r} (None draws fresh entropy)"
        )

    def test_the_default_ceiling_message_is_unchanged(self) -> None:
        assert randomization_seed_error(MAX_EVAL_SEED + 1, "run_policy", max_seed=MAX_EVAL_SEED) == (
            f"run_policy: seed must be an integer in [0, {MAX_EVAL_SEED}] or None, got {MAX_EVAL_SEED + 1} "
            "(a rollout seed is applied to the legacy NumPy global RNG, which refuses a larger value)"
        )


class TestUsableSeedsAreStillApplied:
    """Over-reach control: the refusal is scoped to the one value that has no
    seed to apply."""

    @pytest.mark.parametrize("seed", APPLIER_USABLE_SEEDS, ids=repr)
    def test_a_usable_seed_is_applied_reproducibly(self, seed: int) -> None:
        set_eval_seed(seed)
        first = [random.random() for _ in range(3)]
        set_eval_seed(seed)
        assert [random.random() for _ in range(3)] == first

    @pytest.mark.parametrize("seed", APPLIER_USABLE_SEEDS, ids=repr)
    def test_a_usable_seed_reaches_numpy_too(self, seed: int) -> None:
        np = pytest.importorskip("numpy")
        set_eval_seed(seed)
        first = np.random.random(3).tolist()
        set_eval_seed(seed)
        assert np.random.random(3).tolist() == first

    def test_distinct_seeds_stay_distinct(self) -> None:
        """A refusal that collapsed seeds onto one stream would pass the tests
        above; different seeds must still give different streams."""
        set_eval_seed(7)
        seven = [random.random() for _ in range(3)]
        set_eval_seed(8)
        assert [random.random() for _ in range(3)] != seven
