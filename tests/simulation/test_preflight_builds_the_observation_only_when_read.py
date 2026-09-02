"""The policy preflight builds the runtime observation only when it is read.

``SimEngine._preflight_policy_config`` runs a provider's class-level
:meth:`~strands_robots.policies.base.Policy.preflight` hook against the keys
the runtime observation will carry. Collecting those keys in a simulation is
expensive - ``get_observation`` renders every model camera plus every python
camera in the scene - while
:func:`~strands_robots.policies.preflight_policy` is a no-op for any provider
that does not override the hook, which is every shipped provider except
``lerobot_local``. Building the observation before that decision rendered
every camera and discarded the frames on behalf of ``mock``, the classical
planners (cuRobo, MoveIt2) and the whole-body controllers.

That is not only wasted work. The preflight runs before the rollout loop, so
on a software rasterizer (llvmpipe / ``MUJOCO_GL=osmesa``), where a
whole-scene render is seconds rather than milliseconds, it also delays the
loop's cooperative-stop check - which is what ``stop_policy`` needs to be
observed - by that much per rollout.

So the keys are passed as a supplier and ``preflight_policy`` - the one place
that knows whether the hook consumes them - decides when to invoke it. Pinned
on both sides of that seam: the factory must not ask for keys nobody reads,
and the simulation must not build an observation the factory did not ask for.
Rule 2 of ``test_policy_preflight_fail_fast`` (an unavailable observation
skips the check) is the behaviour the supplier's ``None`` return preserves.
"""

from __future__ import annotations

import pytest

from strands_robots.policies import factory as policy_factory
from strands_robots.policies import preflight_policy, register_policy
from strands_robots.policies.base import Policy
from strands_robots.policies.mock import MockPolicy

_ACCEPTING_PROVIDER = "preflight_accepting_supplier_probe"


class _AcceptingPreflightPolicy(MockPolicy):
    """Overrides ``preflight`` and records the keys each call was given."""

    seen: list[set[str]] = []

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: object) -> None:
        cls.seen.append(set(observation_keys))


@pytest.fixture
def accepting_provider():
    """Register the overriding provider and remove it again after the test."""
    _AcceptingPreflightPolicy.seen.clear()
    register_policy(_ACCEPTING_PROVIDER, lambda: _AcceptingPreflightPolicy)
    try:
        yield _ACCEPTING_PROVIDER
    finally:
        policy_factory._runtime_registry.pop(_ACCEPTING_PROVIDER, None)
        _AcceptingPreflightPolicy.seen.clear()


class _Supplier:
    """Records how often the deferred key supplier was invoked."""

    def __init__(self, keys: set[str] | None) -> None:
        self.keys = keys
        self.calls = 0

    def __call__(self) -> set[str] | None:
        self.calls += 1
        return self.keys


class TestTheFactoryAsksForKeysOnlyWhenTheHookReadsThem:
    """``preflight_policy`` owns the decision, so it owns the invocation."""

    def test_a_provider_with_the_default_no_op_hook_is_never_asked(self):
        """``mock`` inherits ``Policy.preflight``, so nothing reads the keys and
        the cost of collecting them must not be paid."""
        supplier = _Supplier({"joint_0", "camera_top"})
        assert preflight_policy("mock", supplier) is None
        assert supplier.calls == 0

    def test_an_unresolvable_provider_is_never_asked(self):
        """Resolution fails before the hook question can even be asked, so the
        keys are not owed either (``create_policy`` raises the real error)."""
        supplier = _Supplier({"joint_0"})
        assert preflight_policy("nonexistent_provider_xyz_123", supplier) is None
        assert supplier.calls == 0

    def test_an_overriding_provider_is_asked_exactly_once(self, accepting_provider):
        """The one provider family that reads the keys still gets them, whole,
        and pays for exactly one collection."""
        supplier = _Supplier({"joint_0", "camera_top"})
        assert preflight_policy(accepting_provider, supplier) is None
        assert supplier.calls == 1
        assert _AcceptingPreflightPolicy.seen == [{"joint_0", "camera_top"}]

    def test_a_supplier_reporting_no_observation_yet_does_not_run_the_hook(self, accepting_provider):
        """``None`` means the runtime observation is not available, so there is
        nothing to validate the configuration against - the same disposition an
        empty observation has always had, not an empty key set."""
        supplier = _Supplier(None)
        assert preflight_policy(accepting_provider, supplier) is None
        assert supplier.calls == 1
        assert _AcceptingPreflightPolicy.seen == []

    def test_a_plain_key_set_is_still_read(self, accepting_provider):
        """The eager spelling is unchanged: a caller that already holds the keys
        passes them directly."""
        assert preflight_policy(accepting_provider, {"joint_0"}) is None
        assert _AcceptingPreflightPolicy.seen == [{"joint_0"}]

    def test_mock_declares_no_images_and_overrides_no_preflight(self):
        """Premise of the whole rule, and the contract ``requires_images``
        documents: the image-free provider is exactly one that reads no keys."""
        assert MockPolicy(**{}).requires_images is False
        # Compared the way ``preflight_policy`` compares it: accessing a
        # classmethod builds a fresh bound object each time, so identity holds
        # only on the underlying function.
        assert MockPolicy.preflight.__func__ is Policy.preflight.__func__


@pytest.mark.usefixtures("accepting_provider")
class TestTheSimulationBuildsNoObservationTheFactoryDidNotAskFor:
    """The expensive half: a scene with a camera, driven through the real seam."""

    @pytest.fixture
    def counted_sim(self, monkeypatch):
        """A real MuJoCo scene with a camera, counting observation builds."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="preflight_supplier_probe", mesh=False)
        assert sim.create_world()["status"] == "success"
        assert sim.add_robot(name="alice", data_config="so100")["status"] == "success"
        assert sim.add_camera("overhead", position=[0.0, -1.0, 1.0], target=[0.0, 0.0, 0.0])["status"] == "success"
        real = sim.get_observation
        calls: list[str] = []

        def counting(*args, **kwargs):
            calls.append("build")
            return real(*args, **kwargs)

        monkeypatch.setattr(sim, "get_observation", counting)
        yield sim, calls
        sim.cleanup()

    def test_a_no_op_preflight_renders_nothing(self, counted_sim):
        """``mock`` before every ``run_policy`` / ``eval_policy`` /
        ``start_policy`` rollout: no observation, so no camera render."""
        sim, calls = counted_sim
        assert sim._preflight_policy_config("alice", "mock", None) is None
        assert calls == []

    def test_an_overriding_preflight_still_sees_the_camera_keys(self, counted_sim, accepting_provider):
        """Control: the provider that validates camera routing must still get a
        full observation, cameras included - the render is owed here."""
        sim, calls = counted_sim
        assert sim._preflight_policy_config("alice", accepting_provider, None) is None
        assert len(calls) == 1
        assert _AcceptingPreflightPolicy.seen
        assert "overhead" in _AcceptingPreflightPolicy.seen[0]
