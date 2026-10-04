"""
Repro: register_robot(name=<built-in canonical>) silently shadows the curated
built-in with a user stub. End-user sees no warning at default log levels;
Robot(<name>) and sim.add_robot(<name>) then quietly return the stub.

Context: strands-labs/robots v0.5.3 end-user path.

Upstream pin:
  strands_robots/registry/user_registry.py:288-293 — on package-registry
  collision, emits logger.info(...).info is below the default WARNING floor,
  so a user running `python -c "...register_robot(..)"` sees nothing on stderr
  while losing aliases, hardware block, joint_labels, description and joint
  count from the curated entry.

Asymmetry inside the same function:
  user-vs-user collision          -> raises ValueError (line 287)
  user-vs-package collision       -> logger.info (line 289)   <-- the papercut
  user-alias-vs-package-canonical -> raises ValueError via _validate_robots

Pinned by: tests/registry/test_user_registry.py:test_overriding_package_robot_logs_info
Fix proposal (<=15 LOC): escalate to logger.warning for the stderr-visible path,
or (behavioural) refuse unless `overwrite=True` is passed, mirroring the
user-user branch. Behavioural fix couples with 1 test update.

Run:
    python bugbash/register_robot_silent_package_shadow_repro.py
"""

from __future__ import annotations

import logging
import os
import pathlib
import tempfile

# Default log config: WARNING to stderr (what a normal end-user sees).
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    sandbox = tempfile.mkdtemp(prefix="bugbash_user_registry_")
    os.environ["STRANDS_BASE_DIR"] = sandbox
    asset_dir = pathlib.Path(sandbox) / "assets" / "stub"
    asset_dir.mkdir(parents=True)
    (asset_dir / "stub.xml").write_text(
        "<mujoco><worldbody><body/></worldbody></mujoco>"
    )

    # Must import AFTER STRANDS_BASE_DIR is set so the user registry
    # resolver sees the sandbox.
    from strands_robots.registry.loader import invalidate_cache
    from strands_robots.registry.robots import get_robot
    from strands_robots.registry.user_registry import register_robot

    before = get_robot("so101")
    print("BEFORE (curated built-in):")
    print(f"  joints={before.get('joints')}")
    print(f"  description={before.get('description')[:50]!r}")
    print(f"  aliases={before.get('aliases', [])[:3]}")
    print(f"  has hardware block? {bool(before.get('hardware'))}")

    # End-user path: register their own thing with a conflicting name.
    # No `overwrite=True` passed. The only clue is a logger.info call at
    # user_registry.py:289 that is below the default WARNING floor.
    register_robot(
        name="so101",
        model_xml="stub.xml",
        asset_dir="stub",
        description="my test",
        joints=99,
    )

    invalidate_cache()
    after = get_robot("so101")
    print("\nAFTER register_robot(name='so101', ...):")
    print(f"  joints={after.get('joints')}")
    print(f"  description={after.get('description')[:50]!r}")
    print(f"  aliases={after.get('aliases', [])}")
    print(f"  has hardware block? {bool(after.get('hardware'))}")

    print("\nStderr output above this line is what the user saw at WARNING.")
    print("The curated so101 (6-DOF, 3 aliases, hardware config) is gone.")
    print(
        "Any downstream code that relied on so101's joint_labels, aliases, or "
        "hardware driver now silently sees the stub."
    )

    # Document the asymmetry: user-vs-user collision IS a hard refusal.
    try:
        register_robot(
            name="so101", model_xml="stub.xml", asset_dir="stub", description="v2"
        )
    except ValueError as exc:
        print(f"\nAsymmetry: user-vs-user collision raises: {exc}")
    else:
        print("\nUser-vs-user also silent? (unexpected)")


if __name__ == "__main__":
    main()
