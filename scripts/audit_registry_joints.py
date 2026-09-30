#!/usr/bin/env python3
"""Check that every simulated robot's registry ``joints`` is the count the simulation reports.

Why this exists
---------------
``strands_robots/registry/robots.json`` gives every robot a ``joints`` figure.
Three surfaces print it verbatim: ``list_robots()`` (the ``Joints`` column an
agent reads to size an action vector), ``get_robot()``, and the ``N joints`` chip
on every docs robot page. Issue #4147 measured that figure against the loaded
MuJoCo model and found 47 of the 66 simulated robots disagreeing: ``unitree_go2``
said 40 where the model has 12, ``leap_hand`` 41 against 16, ``shadow_hand`` 45
against 24, ``unitree_g1`` 46 against 29, and 24 more were off by one because a
floating base had been counted as a joint. The figure had followed no single
rule (some entries counted raw ``<joint>`` tags, some ``njnt``, some hardware
DOF), so nothing could grade it.

The rule
--------
``joints`` is the number of joints ``Robot(name).get_robot_state()`` reports:
every joint of the model the robot loads (``resolve_model(name)``, the scene
file when the asset ships one) except a floating-base free joint, which the
state reports as ``base`` (position and quaternion) rather than as a joint.
Ball joints and passive joints are reported, so they count; a quadrotor whose
only joint is its free base has ``0``. That is the one count a reader can
confirm from the page's own viewer and from ``get_robot_state``.

Usage
-----
``python scripts/audit_registry_joints.py`` compiles every simulated robot whose
asset is already on disk and prints one row per mismatch; exit status 1 when
any is found, 0 otherwise. ``--download`` fetches the assets that are absent
(the whole corpus on a fresh checkout, several GB). ``--write`` rewrites the
mismatching ``joints`` values in ``robots.json`` in place.
``tests/registry/test_registry_joints_are_the_joints_the_simulation_reports.py``
runs the same comparison per robot and pins the corrected entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO / "strands_robots" / "registry" / "robots.json"


@dataclass(frozen=True)
class Row:
    """One simulated robot's declared and reported joint counts."""

    name: str
    declared: int | None
    reported: int | None
    note: str = ""

    @property
    def mismatch(self) -> bool:
        """Whether the registry disagrees with a model that loaded."""
        return self.reported is not None and self.declared != self.reported


def reported_joint_count(model_path: str) -> int:
    """Count the joints ``get_robot_state`` reports for the model at ``model_path``.

    Every joint except a free joint: the simulation reports a floating base as
    ``base`` rather than as a joint, and reports every hinge, slide and ball
    joint, passive or driven.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    return sum(1 for jnt_id in range(int(model.njnt)) if int(model.jnt_type[jnt_id]) != free)


def simulated_robots(registry: dict[str, dict]) -> list[str]:
    """Registry names that ship a simulation asset, sorted."""
    return sorted(name for name, spec in registry.items() if spec.get("asset"))


def audit(registry: dict[str, dict], *, allow_download: bool = False) -> list[Row]:
    """Compare every simulated robot's ``joints`` with the model it loads.

    A robot whose asset is not on disk (and ``allow_download`` is false) or whose
    model does not compile gets ``reported=None`` with the reason in ``note``; it
    is neither a match nor a mismatch.
    """
    from strands_robots.simulation.model_registry import resolve_model

    rows: list[Row] = []
    for name in simulated_robots(registry):
        declared = registry[name].get("joints")
        path = resolve_model(name, allow_download=allow_download)
        if not path or not Path(path).exists():
            rows.append(Row(name, declared, None, "asset not on disk"))
            continue
        try:
            reported = reported_joint_count(path)
        except Exception as exc:  # noqa: BLE001 - an unloadable asset is reported, not raised
            rows.append(Row(name, declared, None, f"model did not compile: {type(exc).__name__}"))
            continue
        rows.append(Row(name, declared, reported))
    return rows


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    """The registry document as shipped (the ``robots`` mapping lives under that key)."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_corrections(document: dict, rows: Sequence[Row], path: Path = REGISTRY_PATH) -> int:
    """Rewrite every mismatching ``joints`` in place; returns how many changed.

    ``ensure_ascii=True`` keeps every pre-existing escape as it is, so the diff
    touches the corrected lines only.
    """
    changed = 0
    for row in rows:
        if row.mismatch:
            document["robots"][row.name]["joints"] = row.reported
            changed += 1
    if changed:
        path.write_text(json.dumps(document, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    """Print the audit; exit 1 on any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--download", action="store_true", help="fetch assets that are not on disk")
    parser.add_argument("--write", action="store_true", help="rewrite mismatching joints in robots.json")
    args = parser.parse_args(argv)

    document = load_registry()
    rows = audit(document["robots"], allow_download=args.download)
    checked = [r for r in rows if r.reported is not None]
    skipped = [r for r in rows if r.reported is None]
    mismatches = [r for r in rows if r.mismatch]

    print(f"{'robot':<20} {'registry':>8} {'reported':>8}")
    for row in mismatches:
        print(f"{row.name:<20} {row.declared!s:>8} {row.reported!s:>8}")
    for row in skipped:
        print(f"{row.name:<20} {row.declared!s:>8} {'-':>8}  {row.note}")
    print(f"checked {len(checked)} of {len(rows)} simulated robots, {len(mismatches)} mismatches, {len(skipped)} not checked")

    if args.write and mismatches:
        changed = write_corrections(document, mismatches)
        print(f"rewrote {changed} joints values in {REGISTRY_PATH.relative_to(REPO)}")
        return 0
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
