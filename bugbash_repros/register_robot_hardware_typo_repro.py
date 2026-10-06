"""Minimal repro: register_robot silently accepts hardware-field typos.

Target: strands-labs/robots v0.5.3
Rotation: registry_404s
Class: Error UX (asymmetric vs sibling _warn_on_a_near_miss_category)

Symptom
-------
`strands_robots.registry.user_registry.register_robot` runs
`_warn_on_a_near_miss_category(name, category)` at the top of the function
(user_registry.py:311 pre-fix) and emits a WARNING for a near-miss like
``category="arms"`` (meant: ``"arm"``). The same function then accepts an
unknown-key *hardware* block without ANY diagnostic:

    register_robot("myarm", hardware={"driver": "strands", "lerobot_tpye": "koch"})

The typo'd ``lerobot_tpye`` is persisted to ``user_robots.json`` verbatim and
is never read by any downstream consumer (loader checks
``hardware.driver``, ``hardware.lerobot_type``,
``hardware.requires_lerobot_from_source``). If the sibling ``driver="strands"``
carries the asset-less declaration check (user_registry.py:_require_hardware_declaration),
the registration succeeds silently and the user's intent to also bind a
``lerobot_type`` is dropped on the floor.

If instead the declaration check *fails* (e.g. ``driver="lerobot"`` with
``lerobot_tpye="koch"``), the ValueError dumps the full dict verbatim but
still gives no "Did you mean 'lerobot_type'?" hint - while a sibling helper
in the same file 150 lines above (``_warn_on_a_near_miss_category``) uses
``difflib.get_close_matches(cutoff=0.6)`` for exactly this.

Fix: ``_warn_on_unknown_hardware_fields`` (sibling helper), one call site at
the top of ``register_robot`` next to the category warn. +45 LOC.

Run (unfixed): warnings never fire; typo is silent.
Run (fixed):   WARNING "declares hardware.lerobot_tpye=..., did you mean 'lerobot_type'?"
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="sr_hwtypo_")
    os.environ["STRANDS_BASE_DIR"] = tmp
    os.environ["STRANDS_ASSETS_DIR"] = os.path.join(tmp, "assets")
    os.makedirs(os.environ["STRANDS_ASSETS_DIR"], exist_ok=True)

    # Capture WARNING+ to prove sibling behaviour
    stream = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                stream.append((record.levelname, record.getMessage()))

    logging.getLogger().addHandler(CaptureHandler())
    logging.getLogger().setLevel(logging.DEBUG)

    from strands_robots.registry import register_robot, unregister_robot, get_robot

    print("=" * 72)
    print("A) sibling control: category typo fires a WARNING (reference behaviour)")
    print("=" * 72)
    stream.clear()
    e = register_robot(name="ctl_cat", category="arms", hardware={"driver": "strands"})
    cat_warnings = [m for lvl, m in stream if "category" in m]
    print(f"  Entry stored category:      {e.get('category')!r}")
    print(f"  WARNINGs about category:    {len(cat_warnings)}")
    for m in cat_warnings:
        print(f"    > {m}")
    unregister_robot("ctl_cat")

    print()
    print("=" * 72)
    print("B) defect: hardware FIELD typo fires NO warning (asymmetry)")
    print("=" * 72)
    stream.clear()
    # driver='strands' carries the asset-less declaration check, so the typo is silent.
    e = register_robot(name="typo1", hardware={"driver": "strands", "lerobot_tpye": "koch"})
    hw_warnings = [m for lvl, m in stream if "hardware" in m]
    print(f"  Entry stored hardware:      {e.get('hardware')!r}")
    print(f"  WARNINGs about hardware:    {len(hw_warnings)}")
    for m in hw_warnings:
        print(f"    > {m}")

    # And the dead weight persists on disk
    with open(os.path.join(tmp, "user_robots.json")) as f:
        doc = json.load(f)
    stored_hw = doc["robots"]["typo1"]["hardware"]
    print(f"  Disk hardware (verbatim):   {stored_hw!r}")
    print(f"  'lerobot_tpye' persisted:   {'lerobot_tpye' in stored_hw}")
    print(f"  'lerobot_type' intended:    'lerobot_type' not in stored - user intent lost")

    # Downstream readers (lerobot_from_source_entry, factory) ignore the typo'd key
    r = get_robot("typo1")
    print(f"  get_robot('typo1').hardware.get('lerobot_type'): "
          f"{r['hardware'].get('lerobot_type')!r}  <-- None, intent dropped")
    unregister_robot("typo1")

    print()
    print("=" * 72)
    print("C) defect compound: declaration fails - error dumps dict, no did-you-mean")
    print("=" * 72)
    stream.clear()
    try:
        # driver='lerobot' alone does NOT satisfy the asset-less declaration
        # check (needs lerobot_type), so this raises ValueError with the full dict.
        register_robot(name="typo2", hardware={"driver": "lerobot", "lerobot_tpye": "koch"})
    except ValueError as exc:
        msg = str(exc)
        print(f"  Error msg (truncated):      {msg[:140]}...")
        print(f"  Contains 'lerobot_tpye':    {'lerobot_tpye' in msg}  <-- yes, dumped")
        print(f"  Contains 'Did you mean':    {'Did you mean' in msg or 'did you mean' in msg}  <-- in error msg itself")
    hw_warns_c = [m for lvl, m in stream if "hardware" in m]
    print(f"  WARNINGs fired first:       {len(hw_warns_c)}")
    for m in hw_warns_c:
        print(f"    > {m}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    sibling_fires = len(cat_warnings) == 1
    hw_fires = len(hw_warnings) == 1
    print(f"  Sibling category warn fires:        {sibling_fires}")
    print(f"  Hardware-field warn fires:          {hw_fires}")
    print(f"  Symmetric helper coverage:          {sibling_fires and hw_fires}")
    # Repro script semantics: exit 0 when the asymmetry is RESOLVED (sibling+hw both fire).
    # On the UNFIXED branch, hw_fires=False -> exit 1 (defect proven).
    return 0 if (sibling_fires and hw_fires) else 1


if __name__ == "__main__":
    sys.exit(main())
