"""Every ``mesh.tell(...)`` the docs show names the policy the peer runs.

``tell`` sends an ``execute`` command, and the wire boundary refuses an
execute with no ``policy_provider`` - ``validate_command`` raises before the
command leaves the sender. The old quickstart's fleet step once read
``follower.mesh.tell(peer, "hold the tray steady")`` and raised on the very
call it demonstrated. This pins the docs to calls that go through: every
``tell`` a page shows is rebuilt as the ``execute`` it sends and run through
``validate_command``, and a checkpoint it names must be a Hub id the default
allowlist accepts (``docs/learn/mesh/fleet.md`` names ``lerobot/smolvla_base``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strands_robots.mesh.security import ValidationError, validate_command

DOCS = Path(__file__).resolve().parents[1] / "docs"
FLEET = DOCS / "learn" / "mesh" / "fleet.md"

_TELL = re.compile(r"\.tell\((?P<args>(?:[^()]|\([^()]*\))*)\)", re.DOTALL)


def _tell_calls(root: Path = DOCS) -> list[tuple[Path, str]]:
    """Every ``.tell(...)`` under *root*, as ``(page, argument text)`` pairs.

    Generated robot pages carry no mesh prose, so they are skipped.
    """
    calls = []
    for path in sorted(root.rglob("*.md")):
        if root == DOCS and "robots" in path.relative_to(DOCS).parts:
            continue
        for m in _TELL.finditer(path.read_text()):
            calls.append((path, m.group("args")))
    return calls


def _as_execute(args: str) -> dict:
    """Rebuild the ``execute`` command a documented ``tell(peer, instruction, **kw)`` sends."""
    positional = re.findall(r'^\s*"[^"]*"\s*,\s*"(?P<instruction>[^"]*)"', args)
    cmd: dict = {"action": "execute", "instruction": positional[0] if positional else "x"}
    for k, v in re.findall(r'(\w+)=("[^"]*"|[\d.]+)', args):
        cmd[k] = v.strip('"') if v.startswith('"') else float(v)
    return cmd


def test_the_docs_show_tell_at_all():
    assert len(_tell_calls()) >= 2


def test_the_reader_flags_a_tell_that_names_no_provider(tmp_path: Path):
    """The shipped pages all comply, so read a page that does not.

    Without this the reader could match nothing at all and every page would
    pass for want of a case.
    """
    (tmp_path / "bare.md").write_text('mesh.tell(peer, "hold the tray steady")\n')
    (tmp_path / "named.md").write_text('mesh.tell(peer, "x", policy_provider="mock")\n')
    found = {path.name: args for path, args in _tell_calls(tmp_path)}
    assert set(found) == {"bare.md", "named.md"}
    assert "policy_provider=" not in found["bare.md"]
    assert "policy_provider=" in found["named.md"]


@pytest.mark.parametrize("path,args", _tell_calls(), ids=lambda v: v.name if isinstance(v, Path) else "args")
def test_every_documented_tell_names_its_policy_provider(path: Path, args: str):
    assert "policy_provider=" in args, (
        f"{path.relative_to(DOCS)}: mesh.tell({args.strip()!r}) would be refused on the wire"
    )


def test_an_execute_without_a_provider_is_refused_before_it_leaves():
    with pytest.raises(ValidationError, match="policy_provider is required"):
        validate_command({"action": "execute", "instruction": "hold the tray steady"})


@pytest.mark.parametrize("path,args", _tell_calls(), ids=lambda v: v.name if isinstance(v, Path) else "args")
def test_every_documented_tell_goes_through_the_wire_boundary(path: Path, args: str):
    """Rebuilt as the command it sends and validated, so a refused keyword cannot ship as prose."""
    out = validate_command(_as_execute(args))
    assert out["policy_provider"], f"{path.relative_to(DOCS)}: {args.strip()!r}"


def test_the_fleet_page_hands_a_peer_a_hub_checkpoint():
    """The fleet page's ``tell`` names a real provider and a Hub id the default allowlist accepts."""
    text = FLEET.read_text()
    m = re.search(r'\.tell\("arm-b", "stack the cubes",(?P<kw>[^)]*)\)', text, re.DOTALL)
    assert m, "the fleet page no longer tells arm-b to stack the cubes"
    cmd = _as_execute(f'"arm-b", "stack the cubes",{m.group("kw")}')
    out = validate_command(cmd)
    assert out["policy_provider"] == "lerobot_local"
    assert out["pretrained_name_or_path"].startswith("lerobot/")


def test_a_local_checkpoint_path_is_refused_on_the_wire():
    """Why the fleet page names a Hub id and not a checkpoint directory it trained."""
    with pytest.raises(ValidationError, match="not in allowlist"):
        validate_command(
            {
                "action": "execute",
                "instruction": "x",
                "policy_provider": "lerobot_local",
                "pretrained_name_or_path": "/tmp/pick_ckpt",
            }
        )
