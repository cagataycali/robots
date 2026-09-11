# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A provider ``start_task`` cannot build is refused, not raised.

:meth:`~strands_robots.drivers.ur.URDriver.start_task` is the fleet's only
``start_task`` that actually builds a policy from the provider registry -- the
other twelve drivers refuse the verb outright -- so it is the only one where the
build's failure modes reach a caller. It is invoked as an agent tool, where
every verb promises a status envelope and an exception is not something the
caller can handle, and its own docstring documents the answer: "a refusal naming
the provider that could not be built".

The handler named ``(ImportError, TypeError, ValueError)``, which covered
neither half of the real population:

* :func:`~strands_robots.policies.create_policy` documents exactly one
  exception, ``UntrustedRemoteCodeError``, and it is a ``RuntimeError``. Driven
  over every spelling the two policy registries hold, nine of twenty-nine raised
  past the envelope -- ``lerobot_local`` and its ``lerobot`` alias among them,
  the provider a UR arm running a locally trained checkpoint would name, and the
  five remote-code providers escape under the *secure* default with
  ``STRANDS_TRUST_REMOTE_CODE`` unset.
* a provider whose constructor resolves a checkpoint off disk raises
  ``FileNotFoundError`` from a path the caller mistyped, so widening the tuple
  to cover today's classes would have re-broken on the next provider. The
  registry is extensible through
  :func:`~strands_robots.policies.register_policy` as well, so the set is not
  enumerable even in principle.

The pin is therefore the total relation over the whole registry rather than a
list of the nine: a spelling this package registers either builds and rolls out,
or comes back as a refusal that names it. The thirtieth provider is held to it
the day it lands.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from strands_robots.drivers.ur import URDriver
from strands_robots.policies import list_aliases, list_providers
from tests.mocks.ur_rtde import FakeRTDE, text_of

HOST = "192.168.1.10"

#: Every spelling :func:`create_policy` resolves through the registries, which is
#: the population a caller can name in ``policy_provider``.
REGISTERED = sorted(set(list_providers()) | set(list_aliases()))

#: A rollout envelope, returned in place of the real one so a cell grades the
#: build gate and not the control loop behind it.
ROLLED_OUT = {"status": "success", "content": [{"text": "rollout started"}]}


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> URDriver:
    """A connected driver whose rollout is stubbed and whose posture is secure.

    ``STRANDS_TRUST_REMOTE_CODE`` is removed rather than assumed absent: five of
    the registered providers refuse to build without it, and that refusal is the
    default posture this verb has to answer under.
    """
    monkeypatch.delenv("STRANDS_TRUST_REMOTE_CODE", raising=False)
    fake = FakeRTDE()
    control = types.ModuleType("rtde_control")
    receive = types.ModuleType("rtde_receive")
    control.RTDEControlInterface = fake.make_control  # type: ignore[attr-defined]
    receive.RTDEReceiveInterface = fake.make_receive  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rtde_control", control)
    monkeypatch.setitem(sys.modules, "rtde_receive", receive)
    monkeypatch.setattr(URDriver, "run_policy", lambda *a, **k: ROLLED_OUT)
    built = URDriver(tool_name="ur5e", port=HOST)
    assert built.connect_eagerly() is None
    return built


class TestEveryRegisteredProviderComesBackAsAnEnvelope:
    """The relation, over the whole registry."""

    def test_the_population_is_not_empty(self) -> None:
        """Non-vacuity: an empty registry would make every cell below pass."""
        assert len(REGISTERED) >= 20, REGISTERED

    @pytest.mark.parametrize("provider", REGISTERED)
    def test_a_registered_provider_builds_or_is_refused(self, driver: URDriver, provider: str) -> None:
        envelope = driver.start_task("pick up the cube", policy_provider=provider)
        assert envelope["status"] in ("success", "error"), envelope
        if envelope["status"] == "error":
            assert provider in text_of(envelope), text_of(envelope)

    def test_some_of_the_population_really_fails_to_build(self, driver: URDriver) -> None:
        """Non-vacuity: a registry that built everything would grade no refusal."""
        refused = [
            p for p in REGISTERED if driver.start_task("pick up the cube", policy_provider=p)["status"] == "error"
        ]
        assert len(refused) >= 10, refused


class TestTheTwoFailureShapesThatEscaped:
    """The regression: both of these raised out of the verb before the fix."""

    def test_a_remote_code_provider_is_refused_under_the_secure_default(self, driver: URDriver) -> None:
        """``UntrustedRemoteCodeError`` is a ``RuntimeError``, so the tuple missed it."""
        envelope = driver.start_task("pick up the cube", policy_provider="lerobot_local")
        assert envelope["status"] == "error"
        assert "lerobot_local" in text_of(envelope)
        assert "trust_remote_code" in text_of(envelope)

    def test_a_mistyped_checkpoint_path_is_refused(self, driver: URDriver) -> None:
        """``FileNotFoundError`` is an ``OSError``, outside any widened tuple."""
        envelope = driver.start_task(
            "pick up the cube",
            policy_provider="rl",
            checkpoint_dir="/nonexistent/checkpoint",
        )
        assert envelope["status"] == "error"
        assert "rl" in text_of(envelope)


class TestABuildableProviderStillReachesTheRollout:
    """Essence: the refusal did not swallow the happy path."""

    def test_a_provider_that_builds_is_handed_to_run_policy(
        self, monkeypatch: pytest.MonkeyPatch, driver: URDriver
    ) -> None:
        seen: dict[str, Any] = {}

        def record(self: URDriver, policy: Any, **kwargs: Any) -> dict[str, Any]:
            seen["policy"] = policy
            seen.update(kwargs)
            return ROLLED_OUT

        monkeypatch.setattr(URDriver, "run_policy", record)
        envelope = driver.start_task("pick up the cube", policy_provider="mock", duration=1.5)
        assert envelope == ROLLED_OUT
        assert seen["instruction"] == "pick up the cube"
        assert seen["duration"] == 1.5
        assert type(seen["policy"]).__name__ == "MockPolicy"
