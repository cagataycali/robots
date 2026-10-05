"""Repro: add_object(material={'builtin': <typo>}) has no 'Did you mean' hint.

Sibling defect to harness#726 (shape typo) in the same file ---
``strands_robots/simulation/mujoco/spec_builder.py``.

Line 151 of the same file uses ``difflib.get_close_matches`` for *material
KEY* typos ("rgb3" → "did you mean 'rgb2'"), but line 768 (material *value*
``builtin=``) just lists the three supported values with no suggestion. A
user following the project's quickstart (``docs/learn/simulation/worlds-and-objects.md``
fence 1 teaches ``material={"builtin": "checker", ...}``) hits:

    builtin='chekker'  (1-char typo on 'checker') → "supported: checker, flat, gradient."

The branch adds +4 LOC that mirror the same ``difflib`` call used 600
lines earlier in the SAME file (``_describe_material_error``) and 150 lines
later (``_geom_type``, fixed in harness#726).

Run on bugbash branch ``bugbash/add-object-builtin-did-you-mean``:
    pip install -e ".[sim-mujoco]"
    python bugbash_repros/add_object_builtin_no_did_you_mean_repro.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.simulation import create_simulation


TYPO_CASES = [
    # (typo, closest valid builtin)
    ("chekker", "checker"),      # 1-char delete
    ("checer", "checker"),       # 1-char delete
    ("gradients", "gradient"),   # 1-char trail
    ("flt", "flat"),             # 1-char delete
    ("flatt", "flat"),           # 1-char trail
]

NOVEL_CASES = ["solid", "shiny", "polka_dot"]  # should still fall through to bare list


def main() -> int:
    sim = create_simulation("mujoco")
    sim.create_world()

    print()
    print("--- Sibling (material-KEY typo) refusal path — DOES have 'did you mean' ---")
    r = sim.add_object(
        name="sibling",
        shape="box",
        size=[0.02, 0.02, 0.02],
        position=[0.1, 0.0, 0.5],
        material={"builtin": "checker", "rgb3": [0.5, 0.5, 0.5]},
    )
    print(f"  [KEY typo 'rgb3'] status={r['status']}")
    print(f"    text: {r['content'][0]['text']}")

    print()
    print("--- Defect: material-VALUE (builtin=) typo refusal — NO 'did you mean' ---")
    for typo, closest in TYPO_CASES:
        r = sim.add_object(
            name=f"obj_{typo}",
            shape="box",
            size=[0.02, 0.02, 0.02],
            position=[0.1, 0.0, 0.5],
            material={"builtin": typo, "rgb1": [0.5, 0.5, 0.5]},
        )
        text = r["content"][0]["text"]
        has_hint = "Did you mean" in text or "did you mean" in text
        marker = "✓ HINT" if has_hint else "✗ NO HINT"
        print(f"  [{marker}] builtin={typo!r} (closest: {closest!r})")
        print(f"    text: {text}")

    print()
    print("--- Novel (unrelated) values should still fall through to the bare list ---")
    for v in NOVEL_CASES:
        r = sim.add_object(
            name=f"novel_{v}",
            shape="box",
            size=[0.02, 0.02, 0.02],
            position=[0.1, 0.0, 0.5],
            material={"builtin": v, "rgb1": [0.5, 0.5, 0.5]},
        )
        print(f"  [novel {v!r}] text: {r['content'][0]['text']}")

    sim.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
