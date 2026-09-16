"""``stop_policy`` answers once the robot is free, so the caller's next action on it is admitted."""

import re
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation import Simulation  # noqa: E402


def _text(result):
    return " ".join(c["text"] for c in result.get("content", []) if isinstance(c, dict) and "text" in c)


def _json(result):
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


@pytest.fixture
def arm():
    sim = Simulation()
    sim.create_world()
    sim.add_robot("so101")
    try:
        yield sim
    finally:
        sim.cleanup()


def test_start_policy_right_after_stop_policy_is_admitted(arm):
    for _ in range(5):
        started = arm.start_policy(robot_name="so101", policy_provider="mock", duration=5.0)
        assert started["status"] == "success", _text(started)
        stopped = arm.stop_policy("so101")
        assert stopped["status"] == "success"
        assert _text(stopped).startswith("Stopped on 'so101'")
        assert _json(stopped) == {"robot": "so101", "was_running": True, "exited": True}
        # exited=True means the join retired the worker, so its Future is gone
        # from the table by the time the call answers. Read _policy_threads
        # first: _active_policy_robots() prunes as a side effect, which would
        # clear a stale entry before this line could see it.
        assert "so101" not in arm._policy_threads
        assert "so101" not in arm._active_policy_robots()
    # Nothing left in flight after the last stop.
    assert "No policies running" in _text(arm.list_policies_running())


def test_a_second_stop_after_a_joined_stop_says_was_not_running(arm):
    arm.start_policy(robot_name="so101", policy_provider="mock", duration=5.0)
    arm.stop_policy("so101")
    again = arm.stop_policy("so101")
    assert _text(again) == "Was not running on 'so101'"
    assert _json(again) == {"robot": "so101", "was_running": False, "exited": None}


class _BlockedPolicy(MockPolicy):
    """A policy whose inference blocks until released - the stop flag is not read inside it."""

    def __init__(self, gate: threading.Event, **kwargs):
        super().__init__(**kwargs)
        self._gate = gate
        self.entered = threading.Event()

    async def get_actions(self, observation_dict, instruction, **kwargs):
        self.entered.set()
        self._gate.wait()
        return await super().get_actions(observation_dict, instruction, **kwargs)


def test_a_worker_blocked_in_inference_is_reported_still_live(arm, monkeypatch):
    monkeypatch.setattr(type(arm), "_POLICY_STOP_JOIN_TIMEOUT", 0.2)
    gate = threading.Event()
    policy = _BlockedPolicy(gate)
    started = arm.start_policy(robot_name="so101", policy_object=policy, duration=5.0)
    assert started["status"] == "success", _text(started)
    assert policy.entered.wait(5.0)
    t0 = time.monotonic()
    stopped = arm.stop_policy("so101")
    elapsed = time.monotonic() - t0
    assert stopped["status"] == "success"
    assert 0.2 <= elapsed < 2.0
    text = _text(stopped)
    assert text.startswith("Stop requested on 'so101', but its policy worker is still live after 0.2s")
    assert "action='start_policy' on this robot is refused" in text
    assert _json(stopped) == {"robot": "so101", "was_running": True, "exited": False}
    # The refusal the text promised is real until the worker exits.
    refused = arm.start_policy(robot_name="so101", policy_provider="mock", duration=1.0)
    assert refused["status"] == "error"
    gate.set()
    arm._policy_threads["so101"].result(timeout=10)
    admitted = arm.start_policy(robot_name="so101", policy_provider="mock", duration=0.1)
    assert admitted["status"] == "success", _text(admitted)


def test_stop_of_a_blocking_run_policy_has_nothing_to_join(arm):
    # A blocking run_policy is driven on its caller's thread: stop from another
    # thread lowers the flag and reports exited=None.
    seen = {}

    def driver():
        seen["result"] = arm.run_policy(robot_name="so101", policy_provider="mock", duration=5.0)

    th = threading.Thread(target=driver)
    th.start()
    deadline = time.time() + 5.0
    while time.time() < deadline and "so101" not in arm._active_policy_robots():
        time.sleep(0.01)
    stopped = arm.stop_policy("so101")
    th.join(timeout=10)
    assert _json(stopped) == {"robot": "so101", "was_running": True, "exited": None}
    assert _text(stopped) == "Stopped on 'so101'"
    assert seen["result"]["status"] == "success"


_ROW = re.compile(r"^\| `stop_policy\(", re.MULTILINE)


def test_the_api_reference_row_names_the_fields_the_envelope_carries(arm):
    """The row a caller reads before their first call names every key they get back.

    The envelope's keys are read off a real stop, not listed by hand: when this
    surface grew ``exited`` the row still documented ``was_running`` alone, so a
    caller reading the reference could not learn that the answer tells them
    whether the robot is free yet - the whole point of the field. Same for the
    join budget, which is graded against the constant rather than a copied
    number, and the empty-name resolution.
    """
    arm.start_policy(robot_name="so101", policy_provider="mock", duration=5.0)
    keys = set(_json(arm.stop_policy("so101")))
    assert keys == {"robot", "was_running", "exited"}, keys

    doc = (Path(__file__).resolve().parents[2] / "docs" / "api-reference.md").read_text(encoding="utf-8")
    rows = [line for line in doc.splitlines() if _ROW.match(line)]
    assert len(rows) == 1, rows  # a broken parse would make the rule below vacuous
    row = rows[0]
    for key in keys - {"robot"}:
        assert f"`{key}`" in row, (
            f"docs/api-reference.md documents stop_policy without naming {key!r}, a key its json "
            f"block really returns: {row}"
        )
    assert f"{type(arm)._POLICY_STOP_JOIN_TIMEOUT:g} s" in row, (
        f"the row does not name the real join budget ({type(arm)._POLICY_STOP_JOIN_TIMEOUT}s): {row}"
    )
    assert "only rollout in flight" in row, row
