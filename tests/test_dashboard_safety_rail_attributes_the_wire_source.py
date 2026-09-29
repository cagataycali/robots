"""The dashboard's safety rail believes a resume only when it can say who sent it.

``MeshBridge._on_safety`` folded every ``strands/safety/estop`` and ``strands/safety/resume``
body into the fleet lockout as it arrived, with ``by`` read from whatever the body claimed. The
SDK's own safety handlers bind an envelope to the TLS session that carried it (the body's
``source_zid`` must agree with ``sample.source_info.source_id.zid``) and refuse the three
mismatched shapes; the dashboard did none of that, so a forged resume moved a locked fleet's
badge to "e-stop?" and a forged estop wore any operator's name (f031, CWE-345).

Now the dashboard applies the SDK's binding rule. A mismatched envelope of either kind is
dropped and written to the trail. An estop is still applied when the wire carried no identity
(stopping never gets harder), marked unverified. A resume the dashboard cannot attribute does not
touch the lockout: the fleet stays locked until a peer proves it clear, and the trail says why.
An attributed resume still lands on "unknown", never "clear".
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard import safety_state
from strands_robots.dashboard.mesh_bridge import MeshBridge

ZID = "a3f9c2e1d4b58706"


def _sample(kind: str, body: dict[str, Any], *, wire_zid: str | None = None) -> Any:
    sample = mock.MagicMock(spec=["payload", "key_expr", "source_info"])
    sample.payload.to_bytes.return_value = json.dumps(body).encode()
    sample.key_expr = f"strands/safety/{kind}"
    sample.source_info = SimpleNamespace(source_id=SimpleNamespace(zid=wire_zid)) if wire_zid else None
    return sample


@pytest.fixture
def bridge() -> MeshBridge:
    b = MeshBridge(peer_id="dash")
    b._running = True
    return b


def _trail(bridge: MeshBridge, action: str) -> list[dict[str, Any]]:
    return [e for e in bridge.activity_log() if e["source"] == "safety" and e["action"] == action]


# --- estop --------------------------------------------------------------------------------------


def test_an_estop_without_a_wire_identity_still_locks_the_fleet_and_says_it_is_unverified(bridge: MeshBridge) -> None:
    bridge._lockout_proof["arm-1"] = time.time()
    bridge._on_safety(_sample("estop", {"source": "operator-laptop", "t": time.time()}))
    assert bridge._lockout.state == "locked"
    assert bridge._lockout_proof == {}, "an applied e-stop invalidates every earlier proof"
    (entry,) = _trail(bridge, "estop")
    assert entry["ok"] is True
    assert entry["detail"]["attributed"] is False
    assert entry["detail"]["claimed"] == "operator-laptop"
    assert "unverified" in bridge._lockout.reason


def test_a_bound_estop_records_the_session_that_sent_it(bridge: MeshBridge) -> None:
    bridge._on_safety(
        _sample("estop", {"source": "operator-laptop", "source_zid": ZID, "t": time.time()}, wire_zid=ZID)
    )
    assert bridge._lockout.state == "locked"
    (entry,) = _trail(bridge, "estop")
    assert entry["detail"]["attributed"] is True
    assert entry["detail"]["wire_zid"] == ZID
    assert "unverified" not in bridge._lockout.reason


# --- resume -------------------------------------------------------------------------------------


def _locked(bridge: MeshBridge) -> None:
    bridge._lockout = safety_state.apply_event(
        bridge._lockout, kind="estop", data={"source": "op"}, now=time.time() - 30
    )
    assert bridge._lockout.state == "locked"


def test_a_resume_the_dashboard_cannot_attribute_leaves_the_fleet_locked(bridge: MeshBridge) -> None:
    _locked(bridge)
    bridge._lockout_proof["arm-1"] = time.time()
    bridge._on_safety(_sample("resume", {"source": "op", "t": time.time()}))
    assert bridge._lockout.state == "locked", "a resume nobody can vouch for is not a change of state"
    assert "attribute" in bridge._lockout.reason
    assert bridge._lockout_proof == {"arm-1": mock.ANY}, "a refused resume does not touch the proofs either"
    (entry,) = _trail(bridge, "resume_unattributed")
    assert entry["ok"] is False
    assert entry["detail"]["claimed"] == "op"
    assert _trail(bridge, "resume") == []


def test_an_attributed_resume_lands_on_unknown_never_clear(bridge: MeshBridge) -> None:
    _locked(bridge)
    bridge._on_safety(_sample("resume", {"source": "op", "source_zid": ZID, "t": time.time()}, wire_zid=ZID))
    assert bridge._lockout.state == "unknown"
    assert bridge._lockout.state != "clear"
    (entry,) = _trail(bridge, "resume")
    assert entry["ok"] is True and entry["detail"]["wire_zid"] == ZID


def test_the_unattributed_resume_is_told_to_the_page_as_not_applied(bridge: MeshBridge) -> None:
    _locked(bridge)
    frames: list[dict[str, Any]] = []
    with mock.patch.object(bridge, "_emit", side_effect=frames.append):
        bridge._on_safety(_sample("resume", {"source": "op"}))
    (frame,) = [f for f in frames if f.get("type") == "safety"]
    assert frame["kind"] == "resume" and frame["applied"] is False


# --- the three mismatched shapes, both kinds --------------------------------------------------------


@pytest.mark.parametrize("kind", ["estop", "resume"])
@pytest.mark.parametrize(
    ("body_zid", "wire_zid", "shape"),
    [
        ("0000deadbeef", ZID, "cross-session"),
        (ZID, None, "stripped"),
        (None, ZID, "predates"),
    ],
)
def test_a_mismatched_envelope_is_dropped_whatever_it_says(
    bridge: MeshBridge, kind: str, body_zid: str | None, wire_zid: str | None, shape: str
) -> None:
    before = bridge._lockout
    body: dict[str, Any] = {"source": "op", "t": time.time()}
    if body_zid is not None:
        body["source_zid"] = body_zid
    bridge._on_safety(_sample(kind, body, wire_zid=wire_zid))
    assert bridge._lockout == before
    (entry,) = _trail(bridge, f"{kind}_refused")
    assert entry["ok"] is False
    assert entry["detail"]["claimed"] == "op"
    assert _trail(bridge, kind) == []


def test_a_refused_estop_does_not_clear_the_proofs(bridge: MeshBridge) -> None:
    bridge._lockout_proof["arm-1"] = 1.0
    bridge._on_safety(_sample("estop", {"source_zid": "0000deadbeef"}, wire_zid=ZID))
    assert bridge._lockout_proof == {"arm-1": 1.0}


def test_the_binding_rule_is_the_sdks_own() -> None:
    """The three refused shapes and the two accepted ones match ``Mesh._decode_bound_safety_envelope``."""
    from strands_robots.dashboard.mesh_bridge import bind_safety_sample

    assert bind_safety_sample(_sample("estop", {}), {}) == ("unbound", None)
    assert bind_safety_sample(_sample("estop", {}, wire_zid=ZID), {"source_zid": ZID}) == ("bound", ZID)
    assert bind_safety_sample(_sample("estop", {}, wire_zid=ZID), {"source_zid": "0000"})[0] == "refused"
    assert bind_safety_sample(_sample("estop", {}), {"source_zid": ZID})[0] == "refused"
    assert bind_safety_sample(_sample("estop", {}, wire_zid=ZID), {})[0] == "refused"
    assert bind_safety_sample(_sample("estop", {}, wire_zid=ZID), {"source_zid": 12})[0] == "refused"
