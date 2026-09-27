"""The not-read notice describes a policy's actions in that policy's words.

``instruction_not_read_notice`` is the one sentence every task envelope appends
for a policy with ``reads_instruction = False``. It used to spell "Its actions -
a test motion on every joint -" for every such class, which is what
``MockPolicy`` does and what a custom non-reader written from
``docs/reference/policies/custom-policies.md`` does not. The clause now comes from the
class's own ``instruction_free_actions``; a class declaring none gets no clause.
"""

from __future__ import annotations

from strands_robots.policies.base import Policy, instruction_not_read_notice
from strands_robots.policies.mock import MockPolicy


class _Quiet(Policy):
    reads_instruction = False

    def __init__(self) -> None:
        self._keys: list[str] = []

    async def get_actions(self, observation_dict, instruction, **kwargs):  # type: ignore[override]
        return [dict.fromkeys(self._keys, 0.0)]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = robot_state_keys

    @property
    def provider_name(self) -> str:
        return "quiet"


class TestACustomNonReaderIsDescribedAsItself:
    def test_the_completed_notice_names_no_motion_the_class_did_not_declare(self) -> None:
        notice = instruction_not_read_notice(_Quiet())
        assert notice is not None
        assert notice.startswith("Note: _Quiet does not read the instruction.")
        assert "test motion" not in notice
        assert "Its actions were commanded to the robot whatever the task says" in notice

    def test_the_pending_notice_names_no_motion_either(self) -> None:
        notice = instruction_not_read_notice(_Quiet, pending=True)
        assert notice is not None
        assert "test motion" not in notice
        assert "Its actions are commanded to the robot whatever the task says" in notice

    def test_a_class_that_declares_its_actions_is_quoted(self) -> None:
        class _Walker(_Quiet):
            instruction_free_actions = "a standing gait"

        notice = instruction_not_read_notice(_Walker())
        assert notice is not None
        assert "Its actions - a standing gait - were commanded" in notice


class TestTheMockKeepsItsClause:
    def test_the_mock_still_says_test_motion_on_every_joint(self) -> None:
        notice = instruction_not_read_notice(MockPolicy())
        assert notice is not None
        assert "Its actions - a test motion on every joint - were commanded" in notice
