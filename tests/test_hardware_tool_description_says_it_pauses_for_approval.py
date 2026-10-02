"""The real robot's tool description tells the agent what execute/start do before it calls them.

Measured across the lane's agent runs: the first assistant sentence before an
``execute`` was "I'll wave the arm right away!" - then the call paused for an
operator. The 283-character description said nothing about the pause, nothing
about ``duration`` (so agents passed none and got a 30 s budget for a 3 s
wave), and nothing about which provider needs ``policy_port`` (so the default
``groot`` refused and the agent asked the operator for a port). The
description is the one text an agent reads before its first call; it now
carries those three facts and the fact that ``mock`` ignores the instruction.
"""

from __future__ import annotations

import os

import pytest

from strands_robots import Robot
from strands_robots.hardware_robot import COMMAND_ALLOW_ENV
from strands_robots.registry.policies import get_policy_provider, list_policy_providers

# These cells grade the lerobot wrapper, so they spell driver="lerobot": since
# the native default (2026-10-01) a bare so101/so100 real-mode call builds
# FeetechDriver instead.


@pytest.fixture
def spec():
    robot = Robot("so101", mode="real", driver="lerobot", port=os.devnull)
    try:
        yield robot.tool_spec
    finally:
        robot.cleanup()


def _description(spec) -> str:
    return spec["description"]


class TestTheDescription:
    def test_first_sentence_is_the_action(self, spec) -> None:
        assert _description(spec).startswith("Drive the real robot ")

    def test_names_the_approval_pause_and_the_headless_way(self, spec) -> None:
        text = _description(spec)
        assert "pause for operator approval before the arm moves" in text
        assert f"{COMMAND_ALLOW_ENV}=execute,start" in text

    def test_names_duration_and_its_default(self, spec) -> None:
        assert "at most duration seconds (default 30)" in _description(spec)

    def test_names_who_needs_a_port_and_who_does_not(self, spec) -> None:
        text = _description(spec)
        assert "default provider lerobot_local builds in process and needs pretrained_name_or_path" in text
        assert "moveit2 dials a server so it needs policy_port" in text

    def test_says_mock_ignores_the_instruction(self, spec) -> None:
        assert "mock ignores the instruction" in _description(spec)

    def test_stays_short_enough_to_be_read(self, spec) -> None:
        assert len(_description(spec)) < 800

    def test_provider_parameter_carries_the_same_facts(self, spec) -> None:
        prop = spec["inputSchema"]["json"]["properties"]["policy_provider"]
        assert "lerobot_local (default) runs a local checkpoint in process and needs" in prop["description"]
        assert "pretrained_name_or_path" in prop["description"]
        assert "moveit2 (needs policy_port)" in prop["description"]
        assert "ignores the instruction" in prop["description"]
        assert prop["default"] == "lerobot_local"


class TestItNamesWhatEachProviderItMentionsStillNeeds:
    """A provider offered by name must be runnable from the description alone.

    The registry's ``requires`` is the source of truth for what a provider
    cannot build without, so the description is graded against it rather than
    against a copied list: a provider that grows a requirement fails here
    instead of sending the agent into the refusal this description exists to
    prevent. ``port`` is spelled ``policy_port`` in the tool's own schema.

    The population comes from the registry rather than from
    :func:`~strands_robots.policies.list_providers`, whose dict a caller (or an
    earlier test) can extend at runtime with a provider the registry never
    describes - those have no ``requires`` to grade against.
    """

    @staticmethod
    def _mentioned(text: str) -> set[str]:
        return {name for name in list_policy_providers() if name in text}

    @staticmethod
    def _requires(name: str) -> tuple[str, ...]:
        spec = get_policy_provider(name)
        assert spec is not None, f"{name} is listed by the registry but has no entry"
        return tuple(spec.get("requires") or ())

    def test_it_mentions_the_default_and_an_in_process_alternative(self, spec) -> None:
        mentioned = self._mentioned(_description(spec))
        assert {"moveit2", "mock", "lerobot_local"} <= mentioned, mentioned

    def test_every_mentioned_provider_has_its_requirements_named(self, spec) -> None:
        text = _description(spec)
        graded = 0
        for name in sorted(self._mentioned(text)):
            for keyword in self._requires(name):
                spelled = "policy_port" if keyword == "port" else keyword
                assert spelled in text, f"{name} needs {spelled}, which the description never names"
                graded += 1
        assert graded, "no requirement was graded - the registry declared none"

    def test_only_moveit2_asks_for_a_port(self) -> None:
        needs_port = {n for n in list_policy_providers() if "port" in self._requires(n)}
        assert needs_port == {"moveit2"}, needs_port
