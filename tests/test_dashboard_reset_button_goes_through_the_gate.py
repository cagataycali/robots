"""A Reset on every robot card, through the same gate as any command.

``POST /api/robots/{peer_id}/reset`` sends the peer's wire ``reset``. A simulated
peer resets ungated (a child ``<parent>__<robot>`` is routed to its parent world);
a real arm's reset drives every joint to the home pose at once, so it passes the
same gate a task does and needs the browser's confirmation; a peer whose presence
reports a task in flight is refused with 409 before anything is sent. The card's
button is disabled while a rollout runs: Stop is the verb for that.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import agent_motion  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402
from tests._dashboard_bootstrap import configure_bootstrap  # noqa: E402

OWN_PAGE = {"origin": "http://testserver"}


class FakeBridge:
    def __init__(self, peers: dict[str, dict[str, Any]], answer: dict[str, Any] | None = None) -> None:
        now = time.time()
        self.peers = {pid: {"last_seen": now, **p} for pid, p in peers.items()}
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.activity: list[tuple[str, str, dict[str, Any]]] = []
        self.answer = answer if answer is not None else {"status": "success", "content": [{"text": "reset"}]}

    def snapshot(self) -> dict[str, Any]:
        return {"peers": dict(self.peers)}

    async def send_cmd_async(self, target: str, cmd: dict[str, Any], timeout: float = 30.0, **_: Any) -> dict[str, Any]:
        self.sent.append((target, dict(cmd)))
        return dict(self.answer)

    def record_activity(self, source: str, action: str, **fields: Any) -> None:
        self.activity.append((source, action, fields))

    def stop(self) -> None:
        """The server's shutdown hook stops its bridge."""


SIM_CHILD: dict[str, Any] = {"presence": {"robot_type": "sim", "parent": "lane"}, "state": {"joints": {"j1": 0.0}}}
REAL_ARM: dict[str, Any] = {"presence": {"robot_type": "robot", "hw": "feetech"}, "state": {"joints": {"j1": 0.0}}}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setenv("DASHBOARD_SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.delenv(agent_motion.MOTION_ENV, raising=False)
    monkeypatch.delenv(agent_motion.TASK_CONFIRM_ENV, raising=False)

    # Since f002 (#4298) the fresh-install open posture admits the bootstrap proof, not a loopback peer alone.
    headers = {**OWN_PAGE, **configure_bootstrap(monkeypatch)}

    def make(bridge: FakeBridge) -> TestClient:
        app = create_app()
        app.state.bridge = bridge
        return TestClient(app, headers=headers)

    return make


def test_reset_is_a_gated_action_that_reads_as_motion() -> None:
    assert "reset" in agent_motion.GATED_ACTIONS
    verdict = agent_motion.agent_motion_allowed("reset", peer=REAL_ARM, target="arm-1", env={})
    assert verdict["allowed"] is False and verdict["physical"] is True
    assert "home pose" in verdict["reason"] and "Nothing was sent" in verdict["reason"]
    assert agent_motion.agent_motion_allowed("reset", peer=SIM_CHILD, target="lane__so101", env={})["allowed"] is True


def test_a_sim_child_resets_ungated_through_its_parent_world(client) -> None:
    bridge = FakeBridge({"lane__so101": SIM_CHILD})
    with client(bridge) as c:
        r = c.post("/api/robots/lane__so101/reset")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["routed_to"] == "lane"
    # No robot_name: the parent's reset dispatcher reads none, and the wire
    # validator refuses a key no dispatcher reads (#4397).
    assert bridge.sent == [("lane", {"action": "reset"})]


def test_a_real_arm_is_refused_without_the_browsers_confirmation_and_nothing_is_sent(client) -> None:
    bridge = FakeBridge({"arm-1": REAL_ARM})
    with client(bridge) as c:
        r = c.post("/api/robots/arm-1/reset")
    assert r.status_code == 403
    # the server renders an HTTPException's payload under ``error``
    detail = r.json()["error"]
    assert detail["ok"] is False and "home pose" in detail["error"] and "Nothing was sent" in detail["error"]
    assert bridge.sent == []
    assert bridge.activity and bridge.activity[0][1] == "reset" and bridge.activity[0][2]["ok"] is False


def test_a_real_arm_resets_when_the_browser_confirmed(client) -> None:
    bridge = FakeBridge({"arm-1": REAL_ARM})
    with client(bridge) as c:
        assert c.post("/api/robots/arm-1/reset", json={"confirmed": "true"}).status_code == 403, "a string is not a yes"
        r = c.post("/api/robots/arm-1/reset", json={"confirmed": True})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True and r.json()["routed_to"] is None
    assert bridge.sent == [("arm-1", {"action": "reset"})]


@pytest.mark.parametrize("state", ["running", "connecting"])
def test_a_peer_with_a_task_in_flight_is_refused_before_anything_is_sent(client, state: str) -> None:
    presence: dict[str, Any] = dict(SIM_CHILD["presence"])
    peer = {**SIM_CHILD, "presence": {**presence, "task_status": state}}
    bridge = FakeBridge({"lane__so101": peer})
    with client(bridge) as c:
        r = c.post("/api/robots/lane__so101/reset")
    assert r.status_code == 409
    sentence = r.json()["error"]["error"]
    assert state in sentence and "stop it" in sentence and "Nothing was sent" in sentence
    assert bridge.sent == []


def test_a_peers_refusal_comes_back_as_its_own_sentence(client) -> None:
    bridge = FakeBridge({"lane__so101": SIM_CHILD}, answer={"error": "unknown action: 'reset'"})
    with client(bridge) as c:
        r = c.post("/api/robots/lane__so101/reset")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["result"]["error"] == "unknown action: 'reset'"


def test_an_unknown_peer_is_404_before_the_rpc(client) -> None:
    bridge = FakeBridge({})
    with client(bridge) as c:
        assert c.post("/api/robots/ghost/reset").status_code == 404
    assert bridge.sent == []


# -- the card ------------------------------------------------------------------------------------------

from tests._dashboard_frontend import FRONTEND_SRC, requires_node, run_frontend  # noqa: E402


@requires_node
class TestTheCardsResetButton:
    def test_the_button_is_disabled_while_a_task_runs_and_says_why(self) -> None:
        got = run_frontend(
            """
const m = await import('./resetAction.ts')
console.log(JSON.stringify({
  idle: m.resetVerdict({ running: false, busy: false, offline: false }),
  running: m.resetVerdict({ running: true, busy: false, offline: false }),
  busy: m.resetVerdict({ running: false, busy: true, offline: false }),
  offline: m.resetVerdict({ running: false, busy: false, offline: true }),
}))
"""
        )
        assert got["idle"]["enabled"] is True and "home pose" in got["idle"]["title"]
        assert got["running"]["enabled"] is False and "stop it first" in got["running"]["title"]
        assert got["busy"]["enabled"] is False
        assert got["offline"]["enabled"] is False and "heartbeat" in got["offline"]["title"]

    def test_the_answer_shown_is_the_servers_sentence_never_a_guess(self) -> None:
        got = run_frontend(
            """
const m = await import('./resetAction.ts')
console.log(JSON.stringify({
  ok: m.interpretReset({ ok: true, routed_to: 'lane', result: {} }),
  refused: m.interpretReset({ ok: false, result: { error: "unknown action: 'reset'" } }),
  silent: m.interpretReset({ ok: false, result: null }),
  nothing: m.interpretReset(null),
}))
"""
        )
        assert got["ok"] == {"ok": True, "text": "reset to home pose (via lane)"}
        assert got["refused"]["ok"] is False and got["refused"]["text"] == "reset refused: unknown action: 'reset'"
        assert got["silent"]["ok"] is False and got["silent"]["ambiguous"] is True
        assert got["nothing"]["ok"] is False and got["nothing"]["ambiguous"] is True


def test_every_card_offers_reset_next_to_the_run_controls_through_the_task_hook() -> None:
    """The button is in the run form, so the card and the detail view both carry it, wired to useTask.reset."""
    run_form = (FRONTEND_SRC / "components" / "RunForm.tsx").read_text(encoding="utf-8")
    assert "resetVerdict(" in run_form and 'aria-label="reset to home pose"' in run_form
    assert "runRisk(presence).physical ? setResetPending(true) : onReset(false)" in run_form, (
        "a real arm's reset must open the confirm sheet; a sim's goes straight through"
    )
    for component in ("RobotCard.tsx", "RobotDetail.tsx"):
        source = (FRONTEND_SRC / "components" / component).read_text(encoding="utf-8")
        assert "onReset={reset}" in source, f"{component} does not wire the reset to the task hook"
    hook = (FRONTEND_SRC / "lib" / "useTask.ts").read_text(encoding="utf-8")
    assert "/reset`" in hook and "confirmed ? { confirmed: true } : {}" in hook, (
        "the hook must send confirmed only when the sheet was answered"
    )
