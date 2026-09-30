"""The mesh docs name the presence fields, audit events and wire replies the code produces.

Issue #4174: ``fleet.md`` described a presence dict with ``robot`` and
``last_seen`` (the code publishes ``robot_id`` and reports ``age``);
``safety-and-estop.md`` listed ``command_refused`` as the audit row for a command
that arrived under lockout (the code writes ``command_rejected_lockout``, and
``command_refused`` is a different row: a handler that answered with an error)
and quoted a lockout reply ``refused: lockout engaged`` that no code path
produces (the wire reply is ``{"type": "error", "error": "command rejected"}``,
deliberately generic). Each cell here reads the name from the code and looks
for it in the sentence, so the two cannot drift apart again.
"""

from __future__ import annotations

from pathlib import Path

from strands_robots.mesh import security
from strands_robots.mesh.session import PeerInfo

DOCS = Path(__file__).resolve().parents[2] / "docs" / "learn" / "mesh"
FLEET = DOCS / "fleet.md"
SAFETY = DOCS / "safety-and-estop.md"


def _presence_line() -> str:
    return next(line for line in FLEET.read_text(encoding="utf-8").splitlines() if "presence dicts:" in line)


def test_the_presence_comment_names_fields_the_dict_carries() -> None:
    """Every field the fleet page lists is a key of a real presence dict."""
    peer = PeerInfo(peer_id="arm-b", peer_type="sim", hostname="lab", caps={"robot_id": "arm-b", "robot_type": "sim"})
    row = peer.to_dict()
    line = _presence_line()
    listed = [f.strip(" `") for f in line.split("presence dicts:")[1].split(",") if f.strip(" `") not in ("...", "")]
    assert listed, line
    missing = [f for f in listed if f not in row]
    assert not missing, f"fleet.md lists presence fields the dict does not carry: {missing}; it has {sorted(row)}"
    for required in ("robot_id", "age"):
        assert required in listed, f"fleet.md should name {required!r} (the field a fleet view reads)"


def test_the_presence_comment_drops_the_invented_names() -> None:
    """``robot`` and ``last_seen`` were never keys."""
    line = _presence_line()
    assert " robot," not in line and "last_seen" not in line


def test_the_audit_list_names_the_lockout_row_the_code_writes() -> None:
    """``command_rejected_lockout`` is the row for a command under lockout; ``command_refused`` is a handler's error."""
    text = SAFETY.read_text(encoding="utf-8")
    assert "`command_rejected_lockout`" in text
    assert "`command_refused` under lockout" not in text
    # The names come from the module that emits them, not from prose.
    core = (Path(__file__).resolve().parents[2] / "strands_robots" / "mesh" / "core.py").read_text(encoding="utf-8")
    assert '"command_rejected_lockout"' in core and '"command_refused"' in core


def test_the_lockout_sketch_quotes_the_wire_reply() -> None:
    """The reply is ``type: error`` with the generic text ``command rejected``; nothing says ``lockout engaged``."""
    text = SAFETY.read_text(encoding="utf-8")
    assert "'type': 'error', 'error': 'command rejected'" in text
    assert "refused: lockout engaged" not in text
    core = (Path(__file__).resolve().parents[2] / "strands_robots" / "mesh" / "core.py").read_text(encoding="utf-8")
    assert 'LockoutError("command rejected")' in core


def test_the_lockout_paragraph_lists_the_admitted_actions() -> None:
    """The verbs a locked peer still answers are the code's set, no more and no fewer."""
    text = SAFETY.read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("While engaged, a peer answers only"))
    for action in security.LOCKOUT_ADMITTED_ACTIONS:
        assert f"`{action}`" in line, f"{action!r} is admitted under lockout but the page does not list it"
    for action in sorted(security.ALLOWED_ACTIONS - security.LOCKOUT_ADMITTED_ACTIONS):
        assert f"`{action}`" not in line, f"{action!r} is not admitted under lockout but the page lists it"
