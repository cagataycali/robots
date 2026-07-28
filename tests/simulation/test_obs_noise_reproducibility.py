"""Regression tests: a seeded observation-noise stream restarts each episode.

``set_obs_noise(seed=...)`` documents "a reproducible noise stream", but the
generator was created once and then advanced CONTINUOUSLY across ``reset()``. A
reset is the episode boundary every eval loop uses, so only the FIRST episode of
a run was reproducible:

    set_obs_noise(joint_pos_std=0.01, seed=7)
    episode 1 -> [ 1.2e-05, -0.006205, -0.019012, ...]
    reset(); episode 2 -> [-0.000325, -0.015471, -0.001888, ...]   different
    reset(); episode 3 -> [-0.004632, -0.003037, -0.000241, ...]   different again

Worse, each episode's noise depended on how many observations the PREVIOUS
episode happened to consume, so a re-run whose episode 1 rendered a different
number of frames produced different noise in episode 2 - the seed did not pin the
run at all beyond the first episode.

``reset()`` now restarts the stream from the retained seed. An UNSEEDED stream
stays non-reproducible, which is its documented behaviour.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_SEED = 7
_STD = 0.01


def _noise_sequence(sim, n: int = 4) -> list[float]:
    return [round(float(sim.get_observation("panda", skip_images=True)["joint1"]), 9) for _ in range(n)]


@pytest.fixture
def sim():
    s = Simulation(tool_name="obs_noise_reproducibility", mesh=False)
    s.create_world()
    s.add_robot(name="panda")
    yield s
    s.destroy()


def test_seeded_stream_restarts_on_every_reset(sim) -> None:
    """The core defect: episode 2 and 3 must match episode 1."""
    sim.set_obs_noise(joint_pos_std=_STD, seed=_SEED)
    first = _noise_sequence(sim)
    sim.reset()
    second = _noise_sequence(sim)
    sim.reset()
    third = _noise_sequence(sim)

    assert first == second == third, "seeded noise diverged across episodes"


def test_noise_is_actually_applied(sim) -> None:
    """Guard against the fix degenerating into 'no noise at all'."""
    sim.set_obs_noise(joint_pos_std=_STD, seed=_SEED)
    values = _noise_sequence(sim)
    assert any(abs(v) > 1e-9 for v in values), values


def test_two_runs_with_the_same_seed_agree_episode_for_episode(sim) -> None:
    """A re-run must reproduce EVERY episode, not just the first."""
    sim.set_obs_noise(joint_pos_std=_STD, seed=_SEED)
    run_a = []
    for _ in range(3):
        # Vary the per-episode observation count so a continuous generator would
        # desynchronise the following episodes.
        run_a.append(_noise_sequence(sim, n=2))
        _noise_sequence(sim, n=3)
        sim.reset()

    sim.set_obs_noise(joint_pos_std=_STD, seed=_SEED)
    run_b = []
    for _ in range(3):
        run_b.append(_noise_sequence(sim, n=2))
        sim.reset()

    assert run_a == run_b


def test_unseeded_stream_stays_non_reproducible(sim) -> None:
    """An unseeded stream is explicitly not reproducible; do not change that."""
    sim.set_obs_noise(joint_pos_std=_STD)
    first = _noise_sequence(sim)
    sim.reset()
    assert _noise_sequence(sim) != first


def test_reset_without_noise_configured_is_a_noop(sim) -> None:
    assert sim.reset()["status"] == "success"
    # No noise configured -> observations are exact and repeatable.
    assert _noise_sequence(sim) == _noise_sequence(sim)
