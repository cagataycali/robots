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
from strands_robots.registry.policies import get_policy_provider
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
        """``UntrustedRemoteCodeError`` is a ``RuntimeError``, so the tuple missed it.

        The checkpoint is supplied so the trust gate stays the thing graded:
        ``lerobot_local`` requires it, and
        :class:`TestARequiredKeywordIsJudgedBeforeTheBuild` refuses a build with
        no checkpoint before ``create_policy`` is reached at all. The trust
        refusal is raised whether or not a checkpoint is named, so naming one
        changes nothing about the shape under test.
        """
        envelope = driver.start_task(
            "pick up the cube",
            policy_provider="lerobot_local",
            pretrained_name_or_path="lerobot/smolvla_base",
        )
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


#: The providers whose registry entry names a keyword the caller must supply,
#: read from the registry so a new one is held to this the day it lands.
REQUIRING = sorted(
    (name, tuple(get_policy_provider(name)["requires"]))  # type: ignore[index]
    for name in list_providers()
    if (get_policy_provider(name) or {}).get("requires")
)


class TestARequiredKeywordIsJudgedBeforeTheBuild:
    """A provider that cannot act without a keyword is refused before it is built.

    ``requires`` names the keywords a provider cannot be built usefully without,
    and two of them are not enforced by the constructor they are for.
    ``LerobotLocalPolicy`` defaults ``pretrained_name_or_path=""`` and loads
    lazily; ``Gr00tPolicy`` accepts no ``port`` and falls back to a default
    nobody serves. Both therefore *built*, this verb answered ``success``, and
    the rollout it started held a live arm for one step it could never take:
    measured on a fake controller, ``lerobot_local`` with no checkpoint reached
    ``exit_reason="policy"`` / ``steps: 0`` with "No model loaded and no
    pretrained_name_or_path set", and ``groot`` with no port sat at
    ``running=True`` / ``steps: 0`` for ~15 s of a 2 s budget before a
    ``ConnectionError`` to ``tcp://localhost:5555``.

    The real-arm surface gained this check in #3752; this is the same decision
    at the fleet's only other registry build, which is why the domain moved to
    :func:`~strands_robots.registry.policies.policy_requires_error` rather than being
    written twice.
    """

    def test_the_population_is_not_empty(self) -> None:
        """Non-vacuity: nothing below grades anything if no provider requires a keyword."""
        assert REQUIRING, list_providers()

    @pytest.mark.parametrize(("provider", "requires"), REQUIRING)
    def test_a_missing_required_keyword_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, driver: URDriver, provider: str, requires: tuple[str, ...]
    ) -> None:
        """The refusal names the provider and every keyword that was not supplied."""
        monkeypatch.setattr(
            URDriver, "run_policy", lambda *a, **k: pytest.fail("a rollout was started for an unbuildable policy")
        )
        envelope = driver.start_task("pick up the cube", policy_provider=provider)
        assert envelope["status"] == "error"
        text = text_of(envelope)
        assert text.startswith(f"start_task: policy_provider={provider!r} builds its policy from")
        for keyword in requires:
            assert f"{keyword}=..." in text, text
        assert "the rollout would start on a live arm and fail at its first action" in text

    def test_the_port_is_judged_by_the_same_guard(self, driver: URDriver) -> None:
        """``policy_port`` is funnelled into the build kwargs, so one guard covers it.

        Unlike the real-arm surface -- where ``port`` arrives as a named
        parameter and is judged by its own guard -- this verb puts it in the
        kwargs it builds from, so it is judged with the rest.
        """
        text = text_of(driver.start_task("pick up the cube", policy_provider="groot"))
        assert "builds its policy from port" in text
        assert "the port the policy server listens on" in text

    def test_a_supplied_keyword_is_not_refused(self, monkeypatch: pytest.MonkeyPatch, driver: URDriver) -> None:
        """The guard refuses an absence, never a value: the rollout is still reached."""
        reached: list[str] = []

        def record(self: URDriver, *a: Any, **k: Any) -> dict[str, Any]:
            reached.append("yes")
            return ROLLED_OUT

        monkeypatch.setattr(URDriver, "run_policy", record)
        envelope = driver.start_task("pick up the cube", policy_provider="groot", policy_port=5555)
        assert envelope == ROLLED_OUT
        assert reached == ["yes"]

    def test_the_build_is_never_reached(self, monkeypatch: pytest.MonkeyPatch, driver: URDriver) -> None:
        """ "Before the build" is the point: ``create_policy`` is not called at all.

        This is what separates the fix from a nicer message. The old answer came
        out of the build -- or worse, out of the rollout a successful build
        started -- so a guard that ran after it would still have energized a
        rollout for ``lerobot_local``.
        """
        import strands_robots.policies as policies

        monkeypatch.setattr(
            policies, "create_policy", lambda *a, **k: pytest.fail("the policy was built despite a missing keyword")
        )
        assert driver.start_task("pick up the cube", policy_provider="lerobot_local")["status"] == "error"

    @pytest.mark.parametrize(("provider", "requires"), REQUIRING)
    def test_supplying_what_the_refusal_asks_for_clears_the_guard(
        self, driver: URDriver, provider: str, requires: tuple[str, ...]
    ) -> None:
        """The printed remedy is a call this verb accepts -- for every provider.

        A refusal that names a keyword is only correct if supplying that keyword
        changes the answer; otherwise the caller loops on advice that cannot
        work. So the remedy is applied here rather than asserted: each provider
        is called again with exactly what its own message asked for, and the
        guard must not fire a second time. It may still be refused -- the two
        ``lerobot_local`` spellings then meet the remote-code consent gate,
        which names its own remedy -- but not for a keyword that was supplied.

        This is the domain the previous cell grades at one point: ``port``
        travels as the named ``policy_port`` and the rest inside
        ``**policy_kwargs``, so a guard that read only one of the two would pass
        for ``groot`` and refuse ``lerobot_async`` for a checkpoint it was given.
        """
        supplied: dict[str, Any] = {key: "smolvla" if key == "policy_type" else "x" for key in requires}
        port = supplied.pop("port", None) and 5555
        text = text_of(driver.start_task("pick up the cube", policy_provider=provider, policy_port=port, **supplied))
        assert "builds its policy from" not in text, text

    def test_a_provider_that_requires_nothing_is_untouched(self, driver: URDriver) -> None:
        """Non-vacuity in the other direction: the guard is not refusing everything."""
        assert driver.start_task("pick up the cube", policy_provider="mock") == ROLLED_OUT

    def test_an_unknown_provider_is_left_to_the_build(self, driver: URDriver) -> None:
        """A name the registry does not hold is passed over, not judged against a guess.

        The guard reads the named provider's registry entry, so a spelling with
        no entry has no requirements to be missing. Answering for it here would
        replace ``create_policy``'s refusal -- which names the spelling that
        failed and lists the ones that resolve -- with a keyword complaint about
        a provider that does not exist. The real-arm surface pins the same
        relation on the helper directly; this surface pins it through the verb,
        because it is the one that does *not* ignore ``port`` and so would
        otherwise invent "builds its policy from port" for any typo.
        """
        text = text_of(driver.start_task("pick up the cube", policy_provider="no_such_provider"))
        assert "Unknown policy provider: 'no_such_provider'" in text
        assert "builds its policy from" not in text
