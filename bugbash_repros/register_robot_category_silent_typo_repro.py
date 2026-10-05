"""Repro: register_robot(category=<typo>) opens a lonely silo beside the group it meant.

Fire #78, rotation target: ``registry_404s``.

Symptom
-------
``register_robot()`` accepts any string for ``category`` without validation.
A typo like ``"arms"`` instead of the registry's ``"arm"`` is persisted
verbatim, and :func:`list_robots_by_category` groups the robot under
``"arms"`` - a brand-new one-robot silo that renders beside the real
``"arm"`` group in the CLI table and misses the docs filter row's
``data-family="arm"`` button entirely.

Compare the sibling ``hardware.driver`` validation:
:func:`strands_robots.registry.loader._validate_robots` refuses an unknown
driver outright. The analog for ``category`` is missing.

Rules (default = defect: both halves below must hold on main):

1. Typo category ``"arms"`` is accepted silently - no warning, no refusal, no
   "Did you mean" hint.
2. The resulting :func:`list_robots_by_category` output has an ``"arms"``
   group that holds exactly one robot (``my_arm``), separate from the
   package registry's real ``"arm"`` group.

With the fix on this branch, (1) emits a WARNING log naming the typo and
suggesting ``"arm"`` via :func:`difflib.get_close_matches` - same cutoff
``create_policy()`` and ``_unknown_robot_msg`` use for their "Did you mean"
hints. The silo in (2) is still created (callers may deliberately open new
groups, e.g. ``"quadruped"``, ``"roomba"``), so the fix is a warning-only
nudge, not a refusal.

Run::

    python bugbash_repros/register_robot_category_silent_typo_repro.py
    echo exit=$?

On main: exit=1 (defect).  On fix branch: exit=0.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _run() -> int:
    tmpdir = tempfile.mkdtemp(prefix="bugbash_register_robot_category_")
    os.environ["STRANDS_BASE_DIR"] = tmpdir

    # Capture warnings from the user_registry logger so we can assert the
    # presence/absence of a "Did you mean" hint without parsing stderr.
    user_registry_logger = logging.getLogger("strands_robots.registry.user_registry")
    capture = _CaptureHandler()
    user_registry_logger.addHandler(capture)
    # Make sure WARNINGs propagate past library default.
    original_level = user_registry_logger.level
    user_registry_logger.setLevel(logging.WARNING)

    try:
        from strands_robots.registry import register_robot, list_robots_by_category

        # Typo close to the known "arm" group.
        register_robot(
            name="my_arm",
            hardware={"driver": "strands"},
            description="typo category test",
            category="arms",  # typo for "arm"
            joints=6,
        )

        warnings = [r for r in capture.records if r.levelno == logging.WARNING]
        hinted = any(
            r.name == "strands_robots.registry.user_registry"
            and "my_arm" in r.getMessage()
            and "arms" in r.getMessage()
            and "Did you mean" in r.getMessage()
            and "arm" in r.getMessage()
            for r in warnings
        )

        groups = list_robots_by_category()
        silo = groups.get("arms", [])
        silo_names = {r["name"] for r in silo}

        print(f"warnings seen             : {len(warnings)}")
        print(f"'Did you mean: arm' hint  : {hinted}")
        print(f"groups containing 'arms'  : {'yes' if 'arms' in groups else 'no'}")
        print(f"groups containing 'arm'   : {'yes' if 'arm' in groups else 'no'}")
        print(f"'arms' group content      : {sorted(silo_names)}")

        # Defect holds when BOTH halves are true:
        #   (a) no "Did you mean" hint was emitted
        #   (b) the typo silently opened a sibling group
        silo_opened = "arms" in groups and "my_arm" in silo_names
        if not hinted and silo_opened:
            print("\n[FAIL] typo category 'arms' was accepted silently; one-robot silo opened beside real 'arm' group.")
            return 1

        if hinted and silo_opened:
            print("\n[OK] typo category 'arms' emitted a 'Did you mean: arm' hint; silo is still opened (warning, not refusal).")
            return 0

        print("\n[FAIL] unexpected state - hinted={} silo_opened={}".format(hinted, silo_opened))
        return 2

    finally:
        user_registry_logger.removeHandler(capture)
        user_registry_logger.setLevel(original_level)


if __name__ == "__main__":
    sys.exit(_run())
