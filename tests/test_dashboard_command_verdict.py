"""The verdict on a peer's answer to a command -- and to a STOP.

``command_succeeded`` and ``stop_outcome`` decide whether the dashboard says a task is running and
whether the Stop button reports success, and they feed the per-peer counts behind the fleet estop
(``counts["stopped"]``, ``all_stopped``). Before this file, no test referred to either by name.

The theme throughout: a response that ARRIVED is not a confirmation. The wire is layered, and a
refusal can be phrased in at least five different ways depending on which layer refused.
"""

from strands_robots.dashboard.mesh_bridge import command_succeeded, stop_outcome

# --- command_succeeded: every way a response can say no -------------------------------------------


def test_a_clean_response_is_a_success():
    assert command_succeeded({"type": "response", "result": {"ok": True}}) is True
    assert command_succeeded({"result": {"started": True}}) is True
    assert command_succeeded({"result": "done"}) is True, "a non-dict payload is not evidence of failure"


def test_transport_level_refusals():
    assert command_succeeded(None) is False, "no response at all is not a success"
    assert command_succeeded("nope") is False, "nor is something that is not even a dict"
    assert command_succeeded({"type": "error", "error": "no such peer"}) is False
    assert command_succeeded({"error": "timeout waiting for peer"}) is False
    assert command_succeeded({"ok": False, "result": {}}) is False


def test_the_peer_answered_but_refused():
    assert command_succeeded({"result": {"ok": False}}) is False, "peer exposes no run_task"
    assert command_succeeded({"result": {"status": "error"}}) is False
    assert command_succeeded({"result": {"status": "FAILED"}}) is False, "the word is compared case-insensitively"


def test_an_error_inside_the_payload_is_a_refusal_q88():
    """Q88: a tool refuses with a bare ``{"error": ...}`` -- no ok key, no status.

    This shape used to pass as a success, and both callers turned that into a false statement about
    hardware: the task card said "running" above the robot's own readable error, and a STOP answered
    this way was reported as "stopped".
    """
    assert command_succeeded({"result": {"error": "gripper jammed"}}) is False
    assert command_succeeded({"type": "response", "result": {"error": "no policy loaded"}}) is False
    # A peer that wraps its tool result nests it once more. Both depths are real -- the browser's
    # reader has always looked at result.error ?? result.result.error.
    assert command_succeeded({"result": {"result": {"error": "serial port is in use"}}}) is False


def test_an_empty_error_field_is_not_an_error():
    """A payload that carries the KEY with nothing in it must not be read as a refusal: it would
    turn every such success into a failure, and the message shown to the operator would be blank."""
    assert command_succeeded({"result": {"error": ""}}) is True
    assert command_succeeded({"result": {"error": None}}) is True
    assert command_succeeded({"result": {"result": {"error": ""}}}) is True


def test_ok_true_does_not_override_a_refusal_deeper_down():
    """The layers are not ranked by depth -- ANY layer saying no is a no. A response whose envelope
    is cheerful while its payload carries an error is exactly the Q88 case."""
    assert command_succeeded({"ok": True, "result": {"error": "arm is locked out"}}) is False


# --- stop_outcome: three honest states, and never a false "stopped" ------------------------------


def test_a_real_stop():
    assert stop_outcome({"result": {"ok": True}}) == {"state": "stopped", "detail": ""}


def test_silence_is_never_a_stop():
    """ "unstoppable peer" and "peer offline" need different human reactions, so they are different
    states -- and neither of them is "stopped"."""
    assert stop_outcome(None)["state"] == "no_answer"
    assert stop_outcome({"error": "timeout after 5.0s"})["state"] == "no_answer"
    assert "timeout" in stop_outcome({"error": "timeout after 5.0s"})["detail"], "the reason is kept"


def test_a_peer_that_answered_but_did_not_stop():
    v = stop_outcome({"result": {"ok": False, "error": "no stop_task on this peer"}})
    assert v["state"] == "not_stopped"
    assert v["detail"] == "no stop_task on this peer", "the peer's own words, not a generic word"
    assert stop_outcome({"error": "refused"})["detail"] == "refused"
    assert stop_outcome({"result": {"ok": False}})["detail"] == "refused", "a bare refusal still says something"


def test_an_error_in_the_payload_means_it_did_not_stop_q88():
    """The sharpest edge of Q88: this is the control a human reaches for when an arm is doing
    something they do not like, and this exact shape used to answer "stopped"."""
    v = stop_outcome({"result": {"error": "gripper jammed"}})
    assert v["state"] == "not_stopped"
    assert v["detail"] == "gripper jammed"
    assert stop_outcome({"result": {"result": {"error": "bus busy"}}})["state"] == "not_stopped"


def test_a_non_timeout_error_is_a_refusal_not_a_silence():
    """An error that is not a timeout means the peer (or the bridge) ANSWERED. Calling that
    "no_answer" would tell the operator to worry about a peer that is talking to them."""
    assert stop_outcome({"error": "peer refused: locked out"})["state"] == "not_stopped"
