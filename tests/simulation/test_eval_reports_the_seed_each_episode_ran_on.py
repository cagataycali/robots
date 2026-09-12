"""An evaluation reports the seed each episode ran on, on both of its routes.

:meth:`PolicyRunner.evaluate` has two routes - a ``spec`` (delegated to
``_evaluate_with_spec``) and a ``success_fn`` - and both draw one per-episode
seed from the same master RNG, reseed the process with it, and forward it to
``policy.reset``. Only the ``spec`` route reported it
(``test_seed_recorded_in_per_episode_results`` in
``test_policy_runner_benchmark.py`` pins that side), so on the ``success_fn``
route the value was drawn, used, and dropped.

That is the field a caller needs most on the route that has no dense reward and
no ``info``: the payload says episode 3 of 10 failed, and replaying episode 3
alone requires the seed it ran on. Reconstructing it means replicating a private
master RNG and its draw count, which is exactly the private dependency a public
result exists to avoid.

The ``Args:`` entry for ``seed`` also said "Only used when ``spec`` is
provided", contradicting :meth:`SimEngine.eval_policy` - the ``success_fn``-only
facade - which documents that "each per-episode seed is forwarded to
``policy.reset``". The cells below grade the code, so they establish which of the
two docstrings was describing this route.

An unseeded eval is the deliberate exception rather than a gap: it builds no
master RNG at all ("an unseeded eval must not acquire a global RNG side
effect"), so there is no per-episode seed, and ``None`` reports that instead of
naming a seed the episode never ran on.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies import MockPolicy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: Steps each attempt is allowed. Small: what is under test is the reported
#: seed, not the rollout.
_MAX_STEPS = 3
_N_EPISODES = 4
_SEED = 42

_ROBOT_XML = """
<mujoco model="seed_report_probe">
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom name="link1" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="10"/>
  </actuator>
</mujoco>
"""


class SeedReadingPolicy(MockPolicy):
    """The shipped mock policy, reading the seeds ``reset`` is handed.

    Reads the seed it was GIVEN rather than a process-global RNG, so the cells
    below observe what the eval forwarded rather than who else drew from the
    shared stream.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seeds_seen: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        self.seeds_seen.append(seed)
        super().reset(seed=seed)


@pytest.fixture
def sim():
    engine = Simulation()
    yield engine
    engine.destroy()


@pytest.fixture
def arm(tmp_path, sim) -> str:
    sim.create_world()
    robot_xml = tmp_path / "probe.xml"
    robot_xml.write_text(_ROBOT_XML, encoding="utf-8")
    sim.add_robot("arm1", urdf_path=str(robot_xml))
    return "arm1"


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    assert result.get("status") != "error", result
    return next(c["json"] for c in result["content"] if "json" in c)


def _evaluated(sim, robot: str, policy: MockPolicy, *, seed: int | None) -> dict[str, Any]:
    return _metrics(
        sim.eval_policy(
            robot_name=robot,
            policy_object=policy,
            n_episodes=_N_EPISODES,
            max_steps=_MAX_STEPS,
            success_fn=lambda _obs: False,
            seed=seed,
        )
    )


class TestTheSuccessFnRouteReportsTheSeedItRanOn:
    """The route draws a per-episode seed and hands it back."""

    def test_every_episode_record_carries_its_seed(self, sim, arm) -> None:
        policy = SeedReadingPolicy()
        episodes = _evaluated(sim, arm, policy, seed=_SEED)["episodes"]
        missing = [e["episode"] for e in episodes if "seed" not in e]
        assert not missing, (
            f"episodes {missing} report no seed, so a failed attempt out of "
            f"{len(episodes)} cannot be replayed on its own"
        )

    def test_the_reported_seed_is_the_one_the_policy_was_reset_with(self, sim, arm) -> None:
        """The report names the seed the attempt actually ran on.

        A distinct-and-plausible seed per episode would pass a shape check while
        naming values nothing ran on, so this grades against what ``reset``
        received.
        """
        policy = SeedReadingPolicy()
        episodes = _evaluated(sim, arm, policy, seed=_SEED)["episodes"]
        reported = [e["seed"] for e in episodes]
        assert reported == policy.seeds_seen[: len(reported)]
        assert len(set(reported)) == len(reported), f"episodes must not share a seed: {reported}"

    def test_the_seed_is_used_on_this_route_at_all(self, sim, arm) -> None:
        """``seed`` reaches the policy here, not only on the ``spec`` route.

        The over-reach guard for the docstring correction: were the seed truly
        "only used when ``spec`` is provided", ``reset`` would see ``None``.
        """
        policy = SeedReadingPolicy()
        _evaluated(sim, arm, policy, seed=_SEED)
        assert policy.seeds_seen, "policy.reset was never called, so nothing was seeded"
        assert all(isinstance(s, int) for s in policy.seeds_seen), (
            f"the success_fn route forwarded no seed to policy.reset: {policy.seeds_seen}"
        )

    def test_the_same_master_seed_replays_the_same_episode_seeds(self, sim, arm) -> None:
        """Two evals at one seed report the same per-episode seeds.

        This is what makes a reported seed a replay handle rather than a log line.
        """
        first = [e["seed"] for e in _evaluated(sim, arm, SeedReadingPolicy(), seed=_SEED)["episodes"]]
        second = [e["seed"] for e in _evaluated(sim, arm, SeedReadingPolicy(), seed=_SEED)["episodes"]]
        assert first == second

    def test_an_unseeded_evaluation_reports_no_seed_rather_than_a_number(self, sim, arm) -> None:
        """``None``, because an unseeded eval derives no per-episode seed.

        Reporting a number here would name a seed that replays nothing, which is
        worse than reporting the absence.
        """
        policy = SeedReadingPolicy()
        episodes = _evaluated(sim, arm, policy, seed=None)["episodes"]
        assert [e["seed"] for e in episodes] == [None] * len(episodes)
        assert policy.seeds_seen in ([], [None] * len(episodes))
