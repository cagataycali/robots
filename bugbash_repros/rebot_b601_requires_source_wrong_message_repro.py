"""Repro: Robot('rebot_b601', mode='real') error does not surface the
``requires_lerobot_from_source`` registry flag.

The registry (strands_robots/registry/robots.json:150-162 and :453-464) declares
two entries with ``hardware.requires_lerobot_from_source: true``: ``rebot_b601``
and ``bi_rebot_b601``. docs/robots/rebot_b601.md:23 tells the user that this
robot "is not in the PyPI release" and must come from lerobot main.

The registry flag is honored by docs/hooks/robot_pages.py:490 (doc generation)
and tests/test_lerobot_hardware_conformance.py:80 (test skip list) -- but
NO RUNTIME CODE reads it. When a user installs the stable
``strands-robots[lerobot]`` (as the same docs line instructs) and runs the
Robot() call, they land in strands_robots/hardware_robot.py:1390's generic
fallback and see a 14-name list that excludes their robot, with no mention
of the ``from source`` requirement.

Expected
--------
``ValueError: 'rebot_b601_follower' is a lerobot robot type that is not yet
in a PyPI release. Install lerobot from source (pip install
git+https://github.com/huggingface/lerobot) to use it. See
docs/robots/rebot_b601.md.``

Actual (verified on strands-labs/robots@9b12c90)
----------------------------------------------
``ValueError: Unsupported robot type: 'rebot_b601_follower'. Known lerobot
robot types: ['bi_openarm_follower', 'bi_so_follower',
'earthrover_mini_plus', 'hope_jr_arm', 'hope_jr_hand', 'koch_follower',
'lekiwi', 'lekiwi_client', 'omx_follower', 'openarm_follower', 'reachy2',
'so100_follower', 'so101_follower', 'unitree_g1']``

Reaches hardware_robot.py:1390 (the generic bottom fallback). The two
specific-reason branches above it -- ``_native_driver_refusal`` (line 1370)
and ``_other_lerobot_kind_refusal`` (line 1382) -- cover natively-driven and
teleoperator-arm cases; a third branch for ``requires_lerobot_from_source``
is missing.

Reproducibility
---------------
No hardware needed. Fails at config-class resolution; port='/dev/ttyACM0'
is placeholder-only.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot

ROBOTS = [
    ("rebot_b601", "/dev/ttyACM0"),
    ("bi_rebot_b601", None),
    ("seeed_rebot_b601", "/dev/ttyACM0"),  # alias
]

for name, port in ROBOTS:
    print(f"--- Robot({name!r}, mode='real') ---")
    try:
        kw = {"mode": "real"}
        if port:
            kw["port"] = port
        r = Robot(name, **kw)
        print(f"UNEXPECTED SUCCESS: {type(r)}")
        sys.exit(1)
    except ValueError as e:
        msg = str(e)
        # The error names the lerobot type and a 14-name "known" list that
        # excludes the user's robot. It does NOT say "install from source".
        assert "Unsupported robot type" in msg, msg
        assert "Known lerobot robot types" in msg, msg
        assert "from source" not in msg.lower(), (
            "FIX LANDED - error now mentions 'from source': " + msg
        )
        assert "requires_lerobot_from_source" not in msg, (
            "FIX LANDED - error now mentions the registry flag: " + msg
        )
        print(f"REPRO: {msg[:200]}")
        print(f"  → user has no hint that this type needs lerobot-from-source")

print()
print("All three names reproduce the defect.")
print("Fix site: strands_robots/hardware_robot.py:1370-1392 (add third branch).")
