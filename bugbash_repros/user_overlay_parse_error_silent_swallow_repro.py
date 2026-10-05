"""Repro: a syntax error in user_robots.json makes every custom robot invisible with no mention of why.

Symptom
-------
A user hand-edits ``$STRANDS_BASE_DIR/user_robots.json`` (the overlay
``register_robot`` writes to). Introducing a one-character syntax error — a
trailing comma, an unquoted key, a stray character — causes
``parse_user_robots`` to catch ``json.JSONDecodeError``, log at WARNING, and
return ``{}``. The loader proceeds as if the overlay were absent. Every
previously-registered custom robot vanishes from ``get_robot()`` /
``list_robots()`` / ``Robot(...)``.

The refusal a user then gets is the generic "Unknown robot" 404 — it does not
mention the overlay, the fact that it failed to parse, or where to find it.

Upstream
--------
- ``strands_robots/registry/_overlay.py:66-72`` — the swallow:

      except (json.JSONDecodeError, UnicodeDecodeError) as exc:
          logger.warning("Failed to load user registry %s: %s", user_registry_path(), exc)
          return {}

  Note: the sibling guard right next to the loader
  (``loader.py:183-209 _refuse_unfolded_user_keys``) **raises** ``ValueError``
  with the file path and the exact fix on the same class of mistake (a
  structurally-valid but unusable key). The two paths are inconsistent: a
  trailing comma is a strictly worse user mistake than a mis-folded key, and
  it is the one silently discarded.

Expected
--------
Either raise ``ValueError`` naming the file + line (symmetric with
``_refuse_unfolded_user_keys``), or surface the parse error on the
``Robot(...)`` 404 so the user is pointed at the overlay they just broke.

Fix: ~5 LOC. Raise ``ValueError`` from ``parse_user_robots`` on
``JSONDecodeError`` with the file path, line, and column; the loader call site
already catches ``ValueError`` from the sibling guard, so nothing above needs
to change.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import tempfile

# --- Isolate: fresh STRANDS_BASE_DIR with nothing else in it. --------------
bd = pathlib.Path(tempfile.mkdtemp(prefix="sr_overlay_"))
os.environ["STRANDS_BASE_DIR"] = str(bd)

# Register one custom robot. Needs an asset to pass the hardware/model guard.
asset_dir = bd / "my_scout"
asset_dir.mkdir()
(asset_dir / "scout.xml").write_text("<mujoco/>")

from strands_robots.registry import user_registry  # noqa: E402
from strands_robots.registry.loader import invalidate_cache  # noqa: E402
from strands_robots.registry.robots import get_robot  # noqa: E402

user_registry.register_robot(
    name="my_scout",
    model_xml="scout.xml",
    asset_dir=str(asset_dir),
    description="my scout",
    category="mobile",
    joints=4,
)

assert get_robot("my_scout") is not None, "BEFORE edit: custom robot should resolve"
print("BEFORE edit: get_robot('my_scout') -> present ✓")

# --- The user hand-edit: introduce a trailing comma. -----------------------
p = bd / "user_robots.json"
raw = p.read_text()
broken = raw.replace('"joints": 4,', '"joints": 4,,')
assert broken != raw, "injection should produce a syntax error"
p.write_text(broken)
invalidate_cache()

# Only the WARNING log mentions what went wrong.
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s %(name)s] %(message)s")

# --- Silent-wrong: custom robot vanishes without a trace. ------------------
info = get_robot("my_scout")
assert info is None, "AFTER edit: parser swallowed the error - get_robot returns None silently"
print("AFTER edit:  get_robot('my_scout') -> MISSING (silent)")

# What does the user see when they try to USE their robot?
from strands_robots import Robot  # noqa: E402

try:
    Robot("my_scout", mesh=False)
    raise AssertionError("unreachable: should refuse")
except ValueError as exc:
    msg = str(exc)
    print()
    print("Robot('my_scout', mesh=False) refusal:")
    print(f"  type:    {type(exc).__name__}")
    print(f"  message: {msg[:400]}")
    print()
    mentions_overlay = "user_robots.json" in msg
    mentions_parse = (
        "parse" in msg.lower() or "JSON" in msg or "quotes" in msg.lower() or "comma" in msg.lower()
    )
    if mentions_overlay and mentions_parse:
        print("FIXED: refusal names the overlay file and the parse error.")
        print("The user is pointed at the exact file to fix, not the generic 404.")
    else:
        print("BUG REPRODUCES: refusal is the generic 'Unknown robot' 404.")
        print(f"  mentions user_robots.json: {mentions_overlay}")
        print(f"  mentions parse/JSON error: {mentions_parse}")
        print()
        print("The user is told the robot is unknown. The actual problem — one")
        print("character of invalid JSON in a file this process can see — is")
        print("only in a WARNING log line most interactive runs don't surface.")
