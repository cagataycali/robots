"""Repro: `[stretch]` is not declared in pyproject but the user is guided to it.

Fire sequence a new user actually types:

1. `pip install 'strands-robots[stretch]'`

   This succeeds silently -- pip 24+ does not warn on unknown extras, so there
   is no feedback that `[stretch]` is not a declared extra. (`Provides-Extra`
   in the installed distribution's metadata lists 42 names; `stretch` is not
   among them.)

2. The user follows docs/robots/stretch.md:25 (``Robot("stretch", mode="real")``)
   and hits the StretchDriver's import gate.

3. The refusal says::

        the Stretch SDK is not importable (No module named 'stretch_body'). It
        ships on the robot; elsewhere install it with:
        pip install hello-robot-stretch-body

   -- a different install line entirely. The user is sent to a PyPI package
   name (`hello-robot-stretch-body`), bypassing the extras ecosystem the pattern
   taught them to use.

Shipped family of 20 native drivers: 10 cite ``strands-robots[<name>]`` extras
in the same shape (crazyflie, earthrover, rby1, spot, ur, xarm, microduck,
reachy_media's internal checks, foxglove, serial). Stretch, kinova, kuka,
booster and g1 break the pattern -- the last four have vendor-only SDKs with
honest documented explanations in the error message. Stretch is the odd one
out: ``hello-robot-stretch-body`` IS on PyPI, so a ``[stretch]`` extra would
work.

Run: python bugbash_repro/stretch_extra_mismatch_repro.py
"""
from __future__ import annotations

import importlib.metadata as im

# ---- Part 1: Confirm `stretch` is NOT a declared extra -----------------
md = im.metadata("strands-robots")
extras = sorted(md.get_all("Provides-Extra") or [])
assert "stretch" not in extras, (
    f"BUG FIXED! `stretch` is now a declared extra: {extras}"
)
print(f"[OK] Confirmed `[stretch]` is NOT in Provides-Extra ({len(extras)} extras total)")

# ---- Part 2: Confirm StretchDriver IS shipped --------------------------
from strands_robots import drivers as drivers_mod  # noqa: E402

shipped_names = [cls_name for _, cls_name, _ in drivers_mod._SHIPPED_DRIVERS]
assert "StretchDriver" in shipped_names, (
    f"StretchDriver is not in _SHIPPED_DRIVERS: {shipped_names}"
)
print(f"[OK] StretchDriver IS in _SHIPPED_DRIVERS ({len(shipped_names)} total)")

# ---- Part 3: Confirm the error message DOES NOT cite `[stretch]` -------
from strands_robots.drivers.stretch import _resolve_sdk  # noqa: E402

refusal = _resolve_sdk()
assert isinstance(refusal, str), f"SDK is importable in this venv: {refusal!r}"
print(f"\nStretchDriver refusal (verbatim):\n  {refusal!r}\n")

assert "strands-robots[stretch]" not in refusal, (
    "The refusal now mentions `strands-robots[stretch]`; the bug may be fixed."
)
assert "pip install hello-robot-stretch-body" in refusal, (
    f"The refusal shape has changed: {refusal!r}"
)
print("[OK] Refusal points at bare PyPI name, not an extras-shaped install line")

# ---- Part 4: Sibling drivers DO cite `strands-robots[<name>]` ----------
#
# Spot/rby1/xarm/ur/crazyflie all take the extras-shaped install line.
# This block is the asymmetry argument the issue cites verbatim.
import strands_robots.drivers.spot as spot_mod  # noqa: E402

sib_src = (spot_mod.__file__ or "").replace(".pyc", ".py")
with open(sib_src) as f:
    assert "pip install 'strands-robots[spot]'" in f.read(), (
        "SpotDriver's refusal shape has moved; the asymmetry argument needs re-reading."
    )
print("[OK] SpotDriver (sibling) cites `pip install 'strands-robots[spot]'`")

# ---- Part 5: `[all]` meta-extra does NOT fold stretch in ---------------
import tomllib  # noqa: E402

with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
all_extra = pyproject["project"]["optional-dependencies"]["all"]
stretch_mentioned_in_all = any("stretch" in dep.lower() for dep in all_extra)
assert not stretch_mentioned_in_all, (
    f"`[all]` now folds in stretch: {all_extra}"
)
print(
    f"[OK] `[all]` ({len(all_extra)} refs) does NOT fold in stretch -- "
    f"user who runs `pip install 'strands-robots[all]'` still gets a wedged Stretch."
)

print("\n>>> Papercut reproduced: `[stretch]` is a documented ghost extra. <<<")
