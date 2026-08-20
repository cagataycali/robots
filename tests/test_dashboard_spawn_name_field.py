"""Q55: the spawn form's Name field must reach the peer, and both halves must judge it alike.

``/api/devices/spawn`` and ``DeviceManager.spawn`` have always accepted ``peer_id`` -- it is validated
(``validate_peer_id``), remembered in the board's profile, and used as a zenoh key segment, i.e. it is
the name on the card, in the teleop pair, in every log line. No UI ever sent one, so every arm on the
desk was called ``so101-real-<clock%10000>``: a name the operator did not choose, cannot read as "the
left arm", and cannot change afterwards (a peer id is a live key -- renaming means a respawn).

Pinned here, backend side:

* a chosen name is the peer id, verbatim, and lands in the remembered profile;
* an empty/absent one still means "generate one for me" (the old behaviour must not become a refusal);
* the CLIENT's charset rule is the SERVER's charset rule. That is the mirror-discipline this sweep
  keeps re-learning: a browser that accepts what the server refuses only moves the refusal to after
  the button was pressed, which is exactly the class of defect Q48-Q54 removed.
"""

from __future__ import annotations

import json
import pathlib
import re

import strands_robots.dashboard.device_manager as dm
from strands_robots.dashboard.device_manager import DeviceManager, validate_peer_id

FRONTEND = (
    pathlib.Path(dm.__file__).parent / "frontend" / "src" / "lib" / "peerName.ts"
)


class FakeProc:
    _next_pid = 7100

    def __init__(self, *a, **kw):
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.stdout = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def _manager(tmp_path, monkeypatch):
    monkeypatch.setattr(dm.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(dm.threading, "Thread", lambda *a, **kw: type(
        "T", (), {"start": lambda self: None}
    )())
    mgr = DeviceManager(profiles_path=str(tmp_path / "profiles.json"))
    return mgr


def test_a_chosen_name_becomes_the_peer_id_and_is_remembered(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    monkeypatch.setattr(
        dm,
        "scan_serial_ports",
        lambda *a, **kw: [{"device": "/dev/tty.left", "serial_number": "5AB0181806"}],
    )
    out = mgr.spawn("so101", "real", "left-arm", "/dev/tty.left")
    assert out.get("peer_id") == "left-arm", out
    # The name the operator typed is what the fleet, the profile and the key space all carry.
    assert "left-arm" in mgr.robots
    remembered = json.loads(pathlib.Path(mgr.profiles.path).read_text())
    assert any(
        (p or {}).get("peer_id") == "left-arm" for p in remembered.values()
    ), remembered


def test_no_name_still_means_generate_one(tmp_path, monkeypatch):
    """The field is optional: an empty box must not become a refusal."""
    mgr = _manager(tmp_path, monkeypatch)
    out = mgr.spawn("so101", "sim", None)
    assert "error" not in out, out
    assert out["peer_id"].startswith("so101-sim-")


def test_the_form_sends_peer_id_at_all():
    """The defect itself: the payload had no peer_id, so the route's support was unreachable."""
    panel = FRONTEND.parent.parent / "components" / "DevicePanel.tsx"
    src = panel.read_text()
    body = src.split("post('/api/devices/spawn'", 1)[1][:600]
    assert "peer_id" in body, "the spawn payload must carry the chosen name"


def test_client_and_server_agree_on_what_a_name_may_contain():
    """One rule, two languages. A divergence here re-creates the after-the-button refusal."""
    ts = FRONTEND.read_text()
    m = re.search(r"PEER_NAME_RE\s*=\s*/(?P<body>.+?)/\s*$", ts, re.MULTILINE)
    assert m, "peerName.ts must export the mirrored regex literal"
    client = m.group("body")
    server = dm._PEER_ID_RE.pattern
    assert client == server, (
        f"client charset {client!r} != server charset {server!r}: the browser would accept a name "
        "the server refuses (or refuse one it accepts)"
    )


def test_every_name_the_client_accepts_the_server_accepts():
    """Same claim from the other side, on concrete strings rather than a pattern."""
    ok = ["left-arm", "so101.wrist_2:a-B9", "a", "x" * 64]
    bad = ["left arm", "arms/left", "left*", "", "y" * 65]
    for name in ok:
        assert validate_peer_id(name) is None, name
    for name in bad:
        if name == "":
            continue  # empty means "generate one", which the client sends as None
        assert validate_peer_id(name) is not None, name
