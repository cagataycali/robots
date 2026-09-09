"""A user-registry write stores the whole overlay or leaves it untouched.

Every write to ``user_robots.json`` is a read-modify-write of the WHOLE
document: :func:`~strands_robots.registry.user_registry.register_robot` and
:func:`~strands_robots.registry.user_registry.unregister_robot` both load the
overlay, change one entry and store it back. So a write that lands only
partially does not lose the entry being changed - it loses every robot the
overlay held.

Nothing reports that loss. :func:`parse_user_robots` treats an unparseable
overlay as *no user robots*: it logs a warning and the merged registry falls
back to the package ``robots.json`` alone, so previously registered robots are
simply absent and ``get_robot`` calls them unknown.

Two ways the write could land partially, both closed here:

1. ``json.dump`` encodes straight into the stream it is handed, so a value it
   cannot encode raises only *after* a prefix of the new document has already
   replaced the old one. A ``Path`` or a numpy scalar in the caller-supplied
   ``hardware`` dict is such a value, which made a *rejected* registration
   destroy the overlay it was rejected from - the exact outcome the
   ``_assert_registry_still_loads`` gate one line above the write exists to
   prevent (see ``test_registry_lookup_hardening``, which owns the read-time
   half of that guarantee).
2. Writing into the destination itself truncates it first, so any error partway
   through the commit - a full disk, a crash - leaves a prefix on disk rather
   than the previous good document.

The fix serializes the document in full before the destination is opened and
commits the text through a temp sibling plus ``os.replace``. These pins measure
the property, not the mechanism: after a refused or failed write, the overlay
still reads back exactly the robots it held.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from strands_robots.registry import get_robot
from strands_robots.registry.user_registry import (
    _get_user_registry_path,
    _invalidate_cache,
    get_user_robots,
    register_robot,
    unregister_robot,
)

_MINIMAL_MJCF = '<mujoco><worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'


def _register(assets: Path, name: str, **kwargs: object) -> None:
    """Register *name* against a real one-geom MJCF under *assets*."""
    robot_dir = assets / name
    robot_dir.mkdir(parents=True, exist_ok=True)
    (robot_dir / "bot.xml").write_text(_MINIMAL_MJCF)
    kwargs.setdefault("joints", 6)
    register_robot(name=name, model_xml="bot.xml", asset_dir=name, **kwargs)  # type: ignore[arg-type]


def _populate(assets: Path, count: int = 3) -> list[str]:
    """Register *count* robots and return their names."""
    names = [f"fleet_arm_{i}" for i in range(count)]
    for name in names:
        _register(assets, name)
    return names


# A value JSON cannot encode, reaching the overlay through a documented,
# caller-supplied field. ``hardware`` is typed ``dict[str, Any]`` and stored
# verbatim; ``joints`` is stored verbatim too, and a joint count read off an
# array is a numpy integer.
_UNSERIALIZABLE = [
    pytest.param({"hardware": {"port": Path("/dev/ttyACM0")}}, id="a-path-in-hardware"),
    pytest.param({"hardware": {"motor_ids": {1, 2, 3}}}, id="a-set-in-hardware"),
    pytest.param({"joints": np.int64(6)}, id="a-numpy-joint-count"),
]


@pytest.mark.parametrize("bad_kwargs", _UNSERIALIZABLE)
def test_a_refused_registration_changes_nothing_on_disk(tmp_path, bad_kwargs):
    """The overlay a refused registration was refused from still holds its robots."""
    assets = tmp_path / "assets"
    existing = _populate(assets)
    overlay = _get_user_registry_path()
    before = overlay.read_bytes()

    with pytest.raises(ValueError) as excinfo:
        _register(assets, "rejected_arm", **bad_kwargs)

    # The refusal names the store and says the store is intact, so a caller is
    # not left guessing whether the registration half-landed.
    assert str(overlay) in str(excinfo.value)
    assert "unchanged" in str(excinfo.value)
    # The originating encoder error is kept, so the offending type stays visible.
    assert isinstance(excinfo.value.__cause__, TypeError | ValueError)

    assert overlay.read_bytes() == before
    _invalidate_cache()
    assert sorted(get_user_robots()) == existing
    for name in existing:
        assert get_robot(name) is not None, f"{name} was registered and must still resolve"
    # No debris beside the store: a temp sibling holding a rejected document is
    # itself a document a later reader could pick up.
    assert sorted(p.name for p in overlay.parent.glob("user_robots.json*")) == ["user_robots.json"]


def test_a_commit_that_fails_leaves_the_previous_document(tmp_path, monkeypatch):
    """An I/O error during the commit leaves the stored overlay readable.

    The commit being a rename is what makes that possible, so this also pins
    that a rename is where the new document lands: a write aimed at the
    destination itself has already truncated it by the time any error can be
    raised, and there is no commit step left to interrupt.
    """
    assets = tmp_path / "assets"
    existing = _populate(assets)
    overlay = _get_user_registry_path()
    before = overlay.read_bytes()

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="no space left"):
        _register(assets, "fourth_arm")

    assert overlay.read_bytes() == before
    _invalidate_cache()
    assert sorted(get_user_robots()) == existing
    assert sorted(p.name for p in overlay.parent.glob("user_robots.json*")) == ["user_robots.json"]


def test_an_unregister_whose_commit_fails_keeps_every_robot(tmp_path, monkeypatch):
    """The delete path rewrites the whole overlay too, so it gets the same guarantee.

    Removing one robot re-serializes every other one, which is the case where a
    torn write costs the most: the caller asked to lose a single entry.
    """
    assets = tmp_path / "assets"
    existing = _populate(assets)
    overlay = _get_user_registry_path()
    before = overlay.read_bytes()

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="read-only"):
        unregister_robot(existing[0])

    assert overlay.read_bytes() == before
    _invalidate_cache()
    assert sorted(get_user_robots()) == existing


def test_the_stored_document_is_what_a_reader_parses_back(tmp_path):
    """A successful write is unaffected: same JSON text, four-space indent, one trailing newline."""
    assets = tmp_path / "assets"
    names = _populate(assets, count=2)
    overlay = _get_user_registry_path()

    text = overlay.read_text(encoding="utf-8")
    document = json.loads(text)
    assert sorted(document["robots"]) == names
    assert text == json.dumps(document, indent=4) + "\n"


def test_the_overlay_is_never_encoded_into_the_destination():
    """No writer in the module encodes JSON into an open file.

    The property above holds only because serialization completes before the
    destination is touched. ``json.dump`` writes as it encodes, so a single call
    to it anywhere in this module reintroduces the torn write for whichever
    field is added next; ``json.dumps`` returns text a caller commits by itself.
    """
    import strands_robots.registry.user_registry as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    streaming = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dump"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
    ]
    assert streaming == [], "json.dump encodes into the destination; serialize with json.dumps and commit the text"
    # Population check: the module really is the overlay's writer, so an empty
    # result above means "no streaming encoder", not "nothing found here".
    assert "json.dumps(" in source
    assert "os.replace(" in source
