"""Q12 — `/ws/mesh` re-sent unchanged events at ~35 Hz for a fleet of one.

Measured: 176 msgs / 5.1s = 34.7 Hz for ONE arm and ONE client, of which `presence`
re-sent ~6 Hz and `camera_meta` ~10 Hz although neither changes at that rate. With
12 clients attached that is ~420 JSON serializations per second to say nothing new.

The trap that shapes the fix: `useMesh.ts` sets ``last_seen: Date.now(), stale:
false`` on EVERY event, so the client's liveness comes from event *arrival*. Plain
dedupe would paint an idle peer — one that only publishes presence — as dead while
it is alive. So unchanged events are COALESCED to a low rate (still a liveness
tick), and any real content change is forwarded immediately.
"""

from __future__ import annotations

from strands_robots.dashboard.mesh_bridge import COALESCE_HZ, EventCoalescer, _stable_content


def _presence(peer="arm-1", **data):
    return {"type": "presence", "peer_id": peer, "data": {"ok": True, **data}}


# --------------------------------------------------------------------------
# what gets coalesced, and what must never be
# --------------------------------------------------------------------------

def test_an_unchanged_repeat_inside_the_window_is_dropped():
    c = EventCoalescer({"presence": 1.0})
    assert c.allow(_presence(), 100.0) is True
    assert c.allow(_presence(), 100.1) is False
    assert c.allow(_presence(), 100.9) is False


def test_an_unchanged_repeat_still_arrives_as_a_liveness_tick():
    """NOT optional: the client derives staleness from arrival, so an idle peer
    that only publishes presence must keep showing up."""
    c = EventCoalescer({"presence": 1.0})
    c.allow(_presence(), 100.0)
    assert c.allow(_presence(), 101.01) is True


def test_a_real_change_is_never_delayed():
    c = EventCoalescer({"presence": 1.0})
    c.allow(_presence(mode="idle"), 100.0)
    assert c.allow(_presence(mode="running"), 100.01) is True


def test_a_camera_error_appearing_is_forwarded_at_once():
    c = EventCoalescer({"camera_meta": 2.0})
    ev = {"type": "camera_meta", "peer_id": "arm-1", "cam": "top", "data": {"displayable": True}}
    assert c.allow(ev, 100.0) is True
    broke = {**ev, "data": {"displayable": False, "error": "cannot decode"}}
    assert c.allow(broke, 100.01) is True


def test_a_stale_flag_flip_is_a_change_not_a_repeat():
    c = EventCoalescer({"presence": 1.0})
    c.allow(_presence(stale=False), 100.0)
    assert c.allow(_presence(stale=True), 100.02) is True


def test_a_ticking_timestamp_alone_does_not_defeat_coalescing():
    """The reason a naive payload comparison saves nothing here."""
    c = EventCoalescer({"presence": 1.0})
    assert c.allow(_presence(t=1.0), 100.0) is True
    assert c.allow(_presence(t=1.2), 100.1) is False
    assert c.allow(_presence(t=99999.0), 100.2) is False


def test_state_and_safety_are_always_forwarded():
    """state is real telemetry (the joint traces plot it); safety must never wait."""
    c = EventCoalescer()
    for _ in range(50):
        assert c.allow({"type": "state", "peer_id": "arm-1", "data": {"j": 1}}, 100.0) is True
        assert c.allow({"type": "safety", "kind": "estop"}, 100.0) is True
        assert c.allow({"type": "activity", "data": {"x": 1}}, 100.0) is True


def test_cameras_are_tracked_per_camera_not_per_peer():
    c = EventCoalescer({"camera_meta": 2.0})
    top = {"type": "camera_meta", "peer_id": "arm-1", "cam": "top", "data": {"d": 1}}
    wrist = {"type": "camera_meta", "peer_id": "arm-1", "cam": "wrist", "data": {"d": 1}}
    assert c.allow(top, 100.0) is True
    assert c.allow(wrist, 100.0) is True  # a different tile, not a repeat
    assert c.allow(top, 100.1) is False


def test_peers_are_tracked_independently():
    c = EventCoalescer({"presence": 1.0})
    assert c.allow(_presence("arm-1"), 100.0) is True
    assert c.allow(_presence("arm-2"), 100.0) is True


def test_a_returning_peer_is_not_judged_against_its_former_self():
    c = EventCoalescer({"presence": 1.0})
    c.allow(_presence("arm-1"), 100.0)
    c.forget("arm-1")
    assert c.allow(_presence("arm-1"), 100.05) is True


def test_a_rate_of_zero_disables_coalescing_for_that_type():
    c = EventCoalescer({"presence": 0})
    for _ in range(10):
        assert c.allow(_presence(), 100.0) is True


def test_the_measured_saving_on_the_reported_traffic():
    """Replay the ticket's own numbers: 6 Hz presence + 10 Hz camera_meta over 5s."""
    c = EventCoalescer({"presence": 1.0, "camera_meta": 2.0})
    sent = 0
    for i in range(30):           # 6 Hz presence, unchanged but for its timestamp
        sent += c.allow(_presence(t=i), 100.0 + i / 6.0)
    for i in range(50):           # 10 Hz camera_meta on one tile
        ev = {"type": "camera_meta", "peer_id": "arm-1", "cam": "top",
              "data": {"displayable": True, "t": i}}
        sent += c.allow(ev, 100.0 + i / 10.0)
    assert sent <= 18            # was 80
    assert c.stats()["suppressed_pct"] > 75


# --------------------------------------------------------------------------
# the comparison helper
# --------------------------------------------------------------------------

def test_stable_content_ignores_volatile_fields_at_any_depth():
    a = _stable_content({"cams": {"top": {"displayable": True, "t": 1, "fps": 30}}})
    b = _stable_content({"cams": {"top": {"displayable": True, "t": 9, "fps": 12}}})
    assert a == b


def test_stable_content_notices_a_real_difference():
    a = _stable_content({"cams": {"top": {"displayable": True}}})
    b = _stable_content({"cams": {"top": {"displayable": False}}})
    assert a != b


def test_stable_content_survives_unserializable_payloads():
    class Weird:
        def __repr__(self):
            return "weird"

    assert _stable_content({"x": Weird()})  # no exception, some string


def test_key_order_does_not_change_the_content():
    assert _stable_content({"a": 1, "b": 2}) == _stable_content({"b": 2, "a": 1})


def test_the_shipped_defaults_are_conservative():
    """Both defaults must stay well under the mesh's own publish rates, and far
    above zero -- a 0 would mean 'never tick', which is the dead-peer bug."""
    assert 0 < COALESCE_HZ["presence"] <= 2
    assert 0 < COALESCE_HZ["camera_meta"] <= 5
