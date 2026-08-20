"""The chat box was the one surface where a continuable refusal was not continuable.

Every other seam that turns a refusal into a decision calls attach_consent on an error payload (spawn,
task). A refusal produced INSIDE a tool call never touches those: it becomes a toolResult, the model
paraphrases it, and the operator is left to hunt through Settings for a permission the agent just named.
Q80's agent-motion gate is the guard whose refusal appears ONLY there, so this is its last mile.
"""

from __future__ import annotations

import queue

from strands_robots.dashboard.agent_bridge import WSStreamHandler
from strands_robots.dashboard.agent_motion import MOTION_ENV, agent_motion_allowed

ARM = {"presence": {"hw": "so_follower"}}


def _tool_result(text: str, status: str = "error") -> dict:
    return {
        "message": {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "status": status,
                                        "content": [{"text": text}]}}],
        }
    }


def _drain(q: queue.Queue) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get())
    return out


def test_a_refused_tool_call_carries_the_consent_offer():
    refusal = agent_motion_allowed("task", peer=ARM, target="so101-arm-1", env={})["reason"]
    q: queue.Queue = queue.Queue()
    WSStreamHandler(q)(**_tool_result(refusal))
    (ev,) = _drain(q)
    assert ev["type"] == "tool" and ev["status"] == "error"
    assert ev["needs_consent"]["kind"] == "agent_physical_motion"
    assert ev["needs_consent"]["subject"] == "so101-arm-1"
    assert ev["needs_consent"]["env_var"] == MOTION_ENV


def test_classification_reads_the_full_text_not_the_300_char_preview():
    """THE BUG THIS PINS: the identifying env var sits at the END of the refusal, past the preview cut.
    Classifying the truncated string recognises nothing and the sheet never appears."""
    refusal = agent_motion_allowed("task", peer=ARM, target="so101-arm-1", env={})["reason"]
    assert len(refusal) > 300, "the refusal is long; that is the whole point of this test"
    assert MOTION_ENV not in refusal[:300], "if this ever fits, keep classifying the full text anyway"
    q: queue.Queue = queue.Queue()
    WSStreamHandler(q)(**_tool_result(refusal))
    (ev,) = _drain(q)
    assert "needs_consent" in ev
    # The preview stays bounded — the transcript must not grow a wall of text.
    assert len(ev["result_preview"]) == 300


def test_a_successful_result_offers_nothing():
    q: queue.Queue = queue.Queue()
    WSStreamHandler(q)(**_tool_result("Online peers:\n- so101-arm-1: ...", status="success"))
    (ev,) = _drain(q)
    assert "needs_consent" not in ev


def test_an_ordinary_error_is_not_dressed_up_as_a_permission_problem():
    """A refusal that is not continuable must stay a plain error: offering a grant that cannot help
    teaches the operator to click yes on dialogs that do nothing."""
    q: queue.Queue = queue.Queue()
    WSStreamHandler(q)(**_tool_result("task requires target and instruction"))
    (ev,) = _drain(q)
    assert "needs_consent" not in ev
    assert ev["result_preview"].startswith("task requires")


def test_the_other_guards_reach_the_chat_surface_too():
    """This seam is not agent-motion-specific: any continuable refusal a tool returns is offered."""
    q: queue.Queue = queue.Queue()
    WSStreamHandler(q)(**_tool_result(
        "trust: provider loads models with trust_remote_code=True. export STRANDS_TRUST_REMOTE_CODE=1"
    ))
    (ev,) = _drain(q)
    assert ev["needs_consent"]["kind"] == "trust_remote_code"


def test_a_result_with_no_text_block_does_not_explode():
    q: queue.Queue = queue.Queue()
    WSStreamHandler(q)(**{"message": {"role": "user", "content": [
        {"toolResult": {"toolUseId": "t2", "status": "error", "content": [{"json": {"a": 1}}]}}]}})
    (ev,) = _drain(q)
    assert ev["result_preview"] == "" and "needs_consent" not in ev
