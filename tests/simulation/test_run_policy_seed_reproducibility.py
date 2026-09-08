"""Single-rollout reproducibility via ``run_policy(seed=...)``.

A single ``PolicyRunner.run`` / ``SimEngine.run_policy`` rollout drives a
stochastic policy (VLA action-chunk sampling, diffusion noise) from the
process-global RNG. Without an explicit seed, the same scene + policy produces
a different trajectory on every run - so a manipulation policy can grasp on one
run and miss on the next with no scene change. Multi-episode ``evaluate`` already
seeds per episode; this pins the same reproducibility contract for the
single-rollout path.

Regression for the no-grasp-on-re-run report: identical seed -> identical
trajectory; the seed is forwarded to ``policy.reset(seed=...)``.

The policy double reads the seed it was *handed*, not the process-global RNG.
Every seeded rollout surface forwards the seed it applied to
``policy.reset(seed=...)`` - the contract a service-mode policy relies on, since
its sampler runs in a process ``set_eval_seed`` cannot reach - so the double
seeds private generators from that value and reads the shared object zero times.
Reading the global stream directly made this file's replay comparison depend on
the rollout being that object's only reader for the whole rollout, and one stray
draw between two of the policy's queries shifted every action after it, which the
comparison reported as an unapplied seed. What the comparison then stops
observing - that the rollout applied the seed globally at all - is a call, and
``test_the_forwarded_seed_is_the_one_applied_with_set_eval_seed`` counts it as
one.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.base import Policy
from strands_robots.simulation.policy_runner import OnFrame, PolicyRunner, set_eval_seed

from .test_policy_runner import FakeSim

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Generators a policy double may build - each one is a private stream seeded from
# a value the rollout handed it. Reading the module-level generators (``random.``
# / ``np.random.``) is what these tests exist to keep out of a policy double.
_PRIVATE_GENERATORS = ("random.Random", "np.random.default_rng", "np.random.RandomState")


class StochasticPolicy(Policy):
    """Policy whose actions come from the seed the rollout forwarded to it.

    Stands in for a real VLA whose action-chunk sampling is stochastic. The
    per-step draw is recorded so two rollouts can be compared bit-for-bit.
    Also records the seed passed to :meth:`reset` so the forwarding contract
    can be asserted.

    ``reset`` seeds a private Python and a private NumPy generator from the
    forwarded seed - what a service-mode policy does with it - so the draws
    depend on the seed and on nothing else in the process. An unseeded rollout
    forwards no seed and gets fresh entropy, exactly as it did before.
    """

    def __init__(self) -> None:
        self.robot_state_keys: list[str] = []
        self.draws: list[float] = []
        self.reset_seeds: list[int | None] = []
        self._rngs: tuple[random.Random, np.random.Generator] | None = None

    @property
    def provider_name(self) -> str:
        return "stochastic"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = robot_state_keys

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)
        self._rngs = (random.Random(seed), np.random.default_rng(seed))

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        if self._rngs is None:
            # No seed was forwarded, so there is nothing to be reproducible
            # about: draw from fresh entropy, as an unseeded rollout always did.
            self._rngs = (random.Random(), np.random.default_rng())
        py_rng, np_rng = self._rngs
        # Both kinds of generator, because a real policy samples through either.
        draw = py_rng.random() + float(np_rng.random())
        self.draws.append(draw)
        return [{k: draw for k in self.robot_state_keys}]


def _run_once(seed: int | None, on_frame: OnFrame | None = None) -> StochasticPolicy:
    sim = FakeSim()
    policy = StochasticPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.5,
        control_frequency=10.0,  # -> 5 steps
        action_horizon=1,
        fast_mode=True,
        seed=seed,
        on_frame=on_frame,
    )
    assert result["status"] == "success"
    return policy


def test_same_seed_same_trajectory():
    """Two single rollouts at the same seed draw identical action sequences."""
    a = _run_once(seed=1234)
    b = _run_once(seed=1234)

    assert a.draws, "policy produced no action draws"
    assert a.draws == b.draws, (
        "run_policy(seed=...) is not reproducible: identical seed produced "
        f"different trajectories\n  run A: {a.draws}\n  run B: {b.draws}"
    )


def test_different_seed_different_trajectory():
    """Different seeds diverge - the seed actually drives sampling."""
    a = _run_once(seed=1234)
    b = _run_once(seed=5678)
    assert a.draws != b.draws


def test_seed_forwarded_to_policy_reset():
    """The master seed is forwarded to ``policy.reset(seed=...)`` so service-mode
    policies (e.g. a remote VLA server) can re-init their own RNG."""
    policy = _run_once(seed=42)
    assert 42 in policy.reset_seeds


def test_no_seed_leaves_reset_untouched():
    """Default (seed=None) preserves historical behaviour: reset is not forced."""
    policy = _run_once(seed=None)
    assert policy.reset_seeds == [], (
        f"run_policy with no seed should not call policy.reset(seed=...); got {policy.reset_seeds}"
    )


def test_the_forwarded_seed_is_the_one_applied_with_set_eval_seed(monkeypatch):
    """The rollout also applies the seed to the process-global RNGs.

    A policy that samples in-process is reproducible because ``run`` reseeded
    those RNGs, and the trajectory comparisons above no longer observe that: the
    double reads the forwarded seed instead. A reseed is a *call*, so it is
    counted as one - ``set_eval_seed`` is applied with the same value that
    reaches ``policy.reset``, before the policy is ever queried.

    The seam is the namespace ``run`` resolves the name in. Patching the module
    object a fresh ``import`` returns is not the same thing: that resolves
    through the *package attribute*, which a sibling re-importing the runner
    rebinds to a fresh module for the rest of the session, and the spy then
    records nothing while the same cell passes alone.
    """
    seam = PolicyRunner.run.__globals__
    assert seam["set_eval_seed"] is set_eval_seed, "the seam must be the one this file imported"
    applied: list[int] = []

    def spy(seed: int) -> None:
        applied.append(seed)
        set_eval_seed(seed)

    monkeypatch.setitem(seam, "set_eval_seed", spy)
    policy = _run_once(seed=1234)
    assert applied == [1234], f"the rollout did not apply the seed globally: {applied}"
    assert policy.reset_seeds == applied, (
        f"the seed applied globally and the seed forwarded to the policy differ: {applied} vs {policy.reset_seeds}"
    )


@pytest.mark.parametrize("placement", ["between-queries", "inside-set_eval_seed"])
def test_a_stray_global_draw_does_not_change_the_replay(placement, monkeypatch):
    """The global streams are not the rollout's alone to read.

    ``set_eval_seed`` reseeds the *process-global* RNGs and everything else in
    the interpreter draws from those same objects - a thread left by a fixture, a
    library's trace-id generator. A double that read them directly turned one
    stray draw into a different trajectory, so this file could fail for a draw
    the rollout never made, on a branch that touched no RNG code.

    Two placements, because a stray reader is not confined to one window:

    * ``between-queries`` - an ``on_frame`` hook (user telemetry the runner calls
      once per applied control step) draws once mid-rollout.
    * ``inside-set_eval_seed`` - the reseed itself is not atomic: after
      ``random.seed`` it continues into NumPy, torch and CUDA before returning,
      so a reader can draw while the seed is only half applied. A patched
      ``np.random.seed`` that also draws is the deterministic stand-in for that
      window, and it refuses any instrument that reads the global state after
      ``set_eval_seed`` returns rather than the seed it was handed.

    Either way both runs of a seeded pair must replay identically.
    """
    seed = 1234

    def replay(interfere: bool) -> list[float]:
        if placement == "between-queries":
            stray_step = 2

            def on_frame(step_idx: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
                if step_idx == stray_step:
                    random.random()
                    np.random.random()

            return list(_run_once(seed=seed, on_frame=on_frame if interfere else None).draws)

        with monkeypatch.context() as patched:
            if interfere:
                real_seed = np.random.seed

                def seed_and_draw(*args: Any, **kwargs: Any) -> None:
                    real_seed(*args, **kwargs)
                    random.random()
                    np.random.random()

                patched.setattr(np.random, "seed", seed_and_draw)
            return list(_run_once(seed=seed).draws)

    quiet = replay(interfere=False)
    noisy = replay(interfere=True)
    assert quiet, "policy produced no action draws"
    assert quiet == noisy, (
        "a draw the rollout did not make changed what a seeded rollout replayed"
        f"\n  no stray draw: {quiet}\n  stray draw:    {noisy}"
    )


def test_no_policy_double_measures_a_seed_off_a_process_global_rng():
    """Completeness: a policy double reads the seed it is handed, everywhere.

    A policy double is the only way to observe that a rollout seed was applied,
    and every double that reads ``random`` / ``numpy.random`` is exposed to
    whatever else in the process reads them. One fixed double proves nothing
    about the next one, so the property is asserted over the suite: a ``Policy``
    subclass may build a private generator, and may not draw from the shared one.
    """
    offenders: list[str] = []
    for root in ("tests", "tests_integ"):
        for path in sorted((_REPO_ROOT / root).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ClassDef):
                    continue
                if not any("Policy" in ast.unparse(base) for base in node.bases):
                    continue
                for inner in ast.walk(node):
                    if not isinstance(inner, ast.Call):
                        continue
                    called = ast.unparse(inner.func)
                    if called.startswith(("random.", "np.random.", "numpy.random.")) and not called.endswith(
                        _PRIVATE_GENERATORS
                    ):
                        offenders.append(f"{path.relative_to(_REPO_ROOT)}: {node.name} calls {called}()")
    assert not offenders, (
        "a policy double draws from a process-global RNG, so a seed measurement "
        "made with it can be shifted by any other reader in the process; seed a "
        f"private generator ({', '.join(_PRIVATE_GENERATORS)}) from the seed the "
        "rollout forwards to reset() instead:\n  " + "\n  ".join(offenders)
    )
