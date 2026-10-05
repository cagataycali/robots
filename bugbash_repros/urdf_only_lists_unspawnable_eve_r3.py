#!/usr/bin/env python3
"""
bugbash v0.5.3 — rotation ``registry_404s``
Defect class: Silent-wrong listing (public listing vs loader truth)

Symptom
-------
``strands_robots.list_urdf_only()`` and
``strands_robots.registry.discovery.list_urdf_discoverable()`` publicly
advertise ``eve_r3``, but its ``urdf_robots.json`` entry declares
``has_sim: false`` with a ``refusal`` recorded by the sweep ("upstream
clone failed or URDF_PATH missing after import … Halodi/halodi-robot-models
not found"). An end-user who follows the error-path hint at
``strands_robots/robot.py:226-228`` (which names ``list_urdf_only()`` as a
list of spawnable names) and tries ``Robot("eve_r3", mesh=False)`` hits a
``RuntimeError`` at spawn time. Meanwhile ``list_robots(mode='sim')``
*does* filter the entry out — the three listings disagree.

Upstream file:lines (strands-labs/robots @ 4735957)
---------------------------------------------------
  strands_robots/registry/urdf_robots.json:245-250  eve_r3 entry has_sim=false
  strands_robots/registry/discovery.py:264-273      list_urdf_discoverable (unfiltered)
  strands_robots/registry/discovery.py:387-401      list_urdf_only          (unfiltered)
  strands_robots/robot.py:226-228                   error points user at both lists
  strands_robots/registry/robots.py:280-304         list_robots DOES filter via asset check

Run
---
  python bugbash_repros/urdf_only_lists_unspawnable_eve_r3.py
"""
from __future__ import annotations

import os

# Thor env hygiene (SYSTEM_PROMPT scrubbed to keep subprocess layers clean).
_env = {k: v for k, v in os.environ.items() if k != "SYSTEM_PROMPT"}
os.environ.clear()
os.environ.update(_env)

from strands_robots import Robot, list_robots, list_urdf_only
from strands_robots.registry.discovery import (
    list_urdf_discoverable,
    urdf_registry_entry,
)


def main() -> int:
    # 1. Ground truth: the sweep marked eve_r3 as unbuildable, with a
    #    specific refusal (repo-not-found upstream).
    entry = urdf_registry_entry("eve_r3")
    assert entry is not None, "eve_r3 must still be routable to its refusal"
    assert "asset" not in entry, "eve_r3 must declare no asset (has_sim=false)"
    assert "refusal" in entry, "eve_r3 must carry the sweep refusal"
    print(f"[truth]  eve_r3 refusal → {entry['refusal'][:90]!r}…")

    # 2. list_robots(mode='sim') CORRECTLY filters eve_r3 out.
    sim_names = {r["name"] for r in list_robots(mode="sim")}
    assert "eve_r3" not in sim_names, "list_robots(sim) must not advertise eve_r3"
    print(f"[filter] list_robots(mode='sim')     → eve_r3 absent (correct)")

    # 3. The two public URDF listings advertise it.
    urdf_only = list_urdf_only()
    urdf_disc = list_urdf_discoverable()
    only_leaks = "eve_r3" in urdf_only
    disc_leaks = "eve_r3" in urdf_disc
    print(
        f"[public] list_urdf_only()             → eve_r3 "
        f"{'ADVERTISED (defect)' if only_leaks else 'absent (fixed)'}"
    )
    print(
        f"[public] list_urdf_discoverable()     → eve_r3 "
        f"{'ADVERTISED (defect)' if disc_leaks else 'absent (fixed)'}"
    )

    # 4. The user who followed the hint at robot.py:227 and typed the name
    #    they just saw in list_urdf_only() hits a UrdfBuildError.
    try:
        Robot("eve_r3", mesh=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[spawn]  Robot('eve_r3', mesh=False) raised {type(exc).__name__}")
        assert "does not compile" in str(exc) or "does not build" in str(exc)

    # The repro is read as: on main the two [public] lines say "ADVERTISED";
    # on the branch they say "absent (fixed)". list_robots(sim) and the
    # Robot('eve_r3') refusal are both unchanged.
    return 1 if (only_leaks or disc_leaks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
