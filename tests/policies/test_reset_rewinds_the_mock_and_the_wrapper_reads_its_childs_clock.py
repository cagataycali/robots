"""Two contract gaps the instance-level audit found on the reference policies.

**MockPolicy.reset rewinds the sinusoid.** ``MockPolicy`` is the canonical
reference implementation the ``Policy`` ABC points at, and it is deterministic
but not stateless: ``_step`` advances by one chunk per ``get_actions``. It
inherited the no-op ``reset``, so two episodes seeded alike began wherever the
previous one had ended. Measured on the pre-fix tree: ``reset(seed=1)``,
``get_actions``, ``reset(seed=1)``, ``get_actions`` gave two different chunks.
Every wrapper that forwards ``reset`` to a mock inherited the drift
(``CompositePolicy``, ``PersistentPolicy``, ``RemotePolicy`` over a
``PolicyServer(mock)``), which is how the audit saw it four times.

**PersistentPolicy reads its child's clock.** ``control_frequency`` and
``rtc_observed_delay_steps`` are class attributes on ``Policy`` (``None`` until a
setter runs). The wrapper forwards both setters to the wrapped policy and relies
on ``__getattr__`` to forward reads - but ``__getattr__`` only runs when normal
lookup fails, and normal lookup finds the ``None`` on the class. So after
``wrapper.set_control_frequency(50.0)`` the wrapped policy said ``50.0`` and the
wrapper said ``None``, contradicting the ``__getattr__`` comment that names
``control_frequency`` as one of the attributes it forwards.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.policies import MockPolicy
from strands_robots.policies.persistent import PersistentPolicy

KEYS = ["a", "b", "c"]
OBS = {k: 0.0 for k in KEYS}


def _chunk(policy) -> str:
    return json.dumps(policy.get_actions_sync(dict(OBS), "hold still"), sort_keys=True)


class TestMockResetRewinds:
    def test_two_episodes_seeded_alike_start_at_the_same_phase(self) -> None:
        policy = MockPolicy()
        policy.set_robot_state_keys(KEYS)
        policy.reset(seed=1)
        first = _chunk(policy)
        _chunk(policy)  # advance into the episode
        policy.reset(seed=1)
        assert _chunk(policy) == first

    def test_without_reset_the_sinusoid_keeps_advancing(self) -> None:
        # The rewind is the only thing reset does: successive calls still walk
        # the trajectory, which is what makes the mock a motion and not a pose.
        policy = MockPolicy()
        policy.set_robot_state_keys(KEYS)
        assert _chunk(policy) != _chunk(policy)

    def test_reset_takes_no_seed_too(self) -> None:
        # Both spellings the runtime uses; neither raises (the seed is unread).
        policy = MockPolicy()
        policy.reset()
        policy.reset(seed=None)
        assert policy._step == 0


class TestPersistentReadsTheWrappedClock:
    def test_control_frequency_is_the_one_the_setter_forwarded(self) -> None:
        inner = MockPolicy()
        wrapper = PersistentPolicy("mock", policy_object=inner)
        assert wrapper.control_frequency is None
        wrapper.set_control_frequency(50.0)
        assert inner.control_frequency == 50.0
        assert wrapper.control_frequency == 50.0

    def test_rtc_observed_delay_steps_is_the_one_the_setter_forwarded(self) -> None:
        inner = MockPolicy()
        wrapper = PersistentPolicy("mock", policy_object=inner)
        assert wrapper.rtc_observed_delay_steps is None
        wrapper.set_rtc_observed_delay(3)
        assert inner.rtc_observed_delay_steps == 3
        assert wrapper.rtc_observed_delay_steps == 3
        wrapper.set_rtc_observed_delay(None)
        assert wrapper.rtc_observed_delay_steps is None

    def test_the_setters_still_refuse_what_the_abc_refuses(self) -> None:
        wrapper = PersistentPolicy("mock", policy_object=MockPolicy())
        with pytest.raises(ValueError):
            wrapper.set_control_frequency(0)
        with pytest.raises(ValueError):
            wrapper.set_rtc_observed_delay(-1)
