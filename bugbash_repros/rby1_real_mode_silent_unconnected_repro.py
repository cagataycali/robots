"""Repro for: ``Robot("rby1", mode="real", port=...)`` returns an unconnected
driver that silently reports ``get_observation() == {}`` and whose
``describe()`` raises ``AttributeError`` -- the ``[rby1]`` extra is never cited.

End-user path (as a Rainbow Robotics owner would try):

    pip install strands-robots         # NO [rby1] extra
    python

    >>> from strands_robots import Robot
    >>> r = Robot("rby1", mode="real", port="192.168.30.1:50051")
    >>> r.get_observation()
    {}                                   # silent, no message, no refusal
    >>> r.describe()
    AttributeError: 'RBY1Driver' object has no attribute 'describe'

Three things wrong at once:

1. **No construction-time guard**: ``Robot("rby1", mode="real")`` returns a
   bare ``RBY1Driver`` without calling ``connect_eagerly`` (the one spot
   the ``[rby1]`` extra is checked). The user gets a driver instance back
   as if everything is fine.

2. **Silent empty observation**: ``get_observation()`` returns ``dict(self._joints)``
   which is ``{}`` on an unconnected driver (``_joints: dict[str, float] = {}``
   is set in ``__init__``). Sibling calls that go through ``send_action`` /
   ``state`` DO guard with ``"not connected - call connect_eagerly() first"``
   (lines 486, 566, 654 in ``drivers/rby1.py``). The observation path at
   line 504-510 does not. Same shape on ``SpotDriver.get_observation`` at
   ``drivers/spot.py:403-407`` and (by inspection) the full
   ``drivers/{rby1, spot, stretch, g1, reachy}`` family -- the
   observation seam is the asymmetric one.

3. **AttributeError on documented surface**: ``describe()`` is on
   every ``SimEngine`` and the lerobot ``HardwareRobot``, but **not** on
   the native-driver interface (``RBY1Driver``, ``SpotDriver``, ...). A
   README reader who types ``r.describe()`` on their mode="real" rby1
   gets a bare Python ``AttributeError``, not a hint.

The silent empty observation is the worst of the three: an LLM reasoning
over an empty obs will happily plan an action that references no joint,
and ``send_action({})`` will go into the "nothing to command" refusal
(if it ever reaches it) -- the chain of silent returns conceals the
actual root cause (no ``[rby1]`` extra, no network, no connect call).

Fix outline (~15 LOC):

* In ``RBY1Driver.get_observation`` (and sibling drivers): mirror the
  existing ``state``/``send_action`` guard. If ``self._robot is None``,
  return a refusal naming ``connect_eagerly()`` and the ``[rby1]`` extra
  (same message ``_resolve_sdk`` already composes). The payload shape
  returning plain dict is the public contract of the method -- the clean
  fix is to raise or to return a sentinel; either way, ``{}`` must stop
  reaching the agent as a legitimate observation.

* Add ``describe()`` to the driver protocol (or the ``Robot(mode="real")``
  wrapper), returning something like
  ``{"status": "success", "content": [{"text": "RBY1 driver for 'rby1' @ 192.168.30.1:50051 (not connected; call connect_eagerly())"}]}``.
"""

from __future__ import annotations

import os
import sys

# Must come BEFORE the import so Robot's default MUJOCO_GL path is honoured.
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot


def main() -> int:
    """Reproduce the three silent surfaces on a pip-installed strands-robots
    (no ``[rby1]`` extra, no network to a real RB-Y1)."""
    try:
        r = Robot("rby1", mode="real", port="192.168.30.1:50051")
    except Exception as exc:  # noqa: BLE001
        print(f"UNEXPECTED REFUSAL at construction: {type(exc).__name__}: {exc}")
        return 1

    print(f"Robot built: type={type(r).__name__}")
    print(f"  (expected: a loud refusal citing [rby1] or no network answer)")
    print()

    # Silent-wrong seam #1: get_observation() returns {}
    obs = r.get_observation()
    print(f"get_observation() -> {obs!r}  (type={type(obs).__name__}, len={len(obs)})")
    if obs == {}:
        print("  SILENT-WRONG: empty observation, no status=error, no exception,")
        print("                no mention of [rby1] extra, no connect_eagerly hint.")
    print()

    # Silent-wrong seam #2: describe() raises AttributeError
    try:
        d = r.describe()
        print(f"describe() -> {d!r}")
    except AttributeError as exc:
        print(f"describe() -> AttributeError: {exc}")
        print("  BARE AttributeError on documented end-user surface")
        print("  (SimEngine has describe(); HardwareRobot has describe();")
        print("   native-driver Robot(mode='real') does not).")
    print()

    # The ONE place the [rby1] extra is cited: connect_eagerly().
    err = r.connect_eagerly()
    print(f"connect_eagerly() -> {err!r}")
    if err and "[rby1]" in err:
        print("  OK: connect_eagerly cites the extra. "
              "But a new user hits get_observation first (no reason to")
        print("  know connect_eagerly is a separate call).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
