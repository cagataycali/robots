"""Repro: download_gsplat_scene() has no 'Did you mean' hint.

Defect class: Error UX (asymmetric guidance across sibling refusals).

Observable behaviour
--------------------
``strands_robots.rendering.download_gsplat_scene(name)`` is one of the four
documented entry points on the public rendering surface (re-exported from
``strands_robots.rendering.__init__``). Its four preset keys contain
whitespace::

    'tabletop (indoor room)'
    'bonsai (indoor tabletop)'
    'bicycle (outdoor)'
    'stump (outdoor)'

The *module itself* keys per-scene state by the first-token slug (see
``GSPLAT_SKYBOX_ALIGN`` initialised with ``n.split(" ")[0]`` on
backgrounds.py:802, and the cache-filename slug on backgrounds.py:937) --
"tabletop" and "bonsai" are forms the code already understands. But the
refusal at backgrounds.py:929::

    raise KeyError(f"Unknown scene {name!r}. Known: {list(GSPLAT_SCENES)}")

has no ``difflib`` hint. Every realistic typo -- the bare slug, a
one-character typo of either half, a reordering -- lands on the same
stone-wall "Known: [...]" dump.

Meanwhile 16+ sibling refusals in the same package (``add_object`` shape
guard in isaac/simulation.py:4214, mujoco/spec_builder's shape and material
guards -- see harness#726, harness#749 -- and the 14 sites that use the
``close_match_hint`` helper from utils.py) do use ``difflib.get_close_matches``
and emit "Did you mean X?". ``difflib`` isn't even imported in
backgrounds.py: the hint path simply isn't wired.

Why it bites
------------
A user reading the ``gsplat_scene_names()`` docstring (same file, line 760)
sees ``bonsai`` and ``tabletop`` named in prose; typing them as the arg is
the natural first attempt. The current refusal doesn't tell them to add the
parenthetical.

Run
---
    python3 bugbash_repros/download_gsplat_no_did_you_mean_repro.py

Expected (after fix): each refusal names the closest preset. Before fix:
every refusal is the bare "Unknown scene ... Known: [...]" dump.
"""

from __future__ import annotations

import tempfile

from strands_robots.rendering import GSPLAT_SCENES, download_gsplat_scene

ATTEMPTS = [
    "tabletop",               # bare slug -- the form the module uses in GSPLAT_SKYBOX_ALIGN
    "bonsai",                 # bare slug
    "tabletp (indoor room)",  # one-char typo of first token
    "stupm",                  # one-char typo of slug
    "indoor tabletop",        # paren contents without the key
    "not-a-scene",            # unrelated -- fix MUST NOT emit a hint here
]


def main() -> None:
    print("Known presets:", list(GSPLAT_SCENES))
    print()
    with tempfile.TemporaryDirectory() as td:
        for attempt in ATTEMPTS:
            try:
                download_gsplat_scene(attempt, cache_dir=td, timeout=5)
            except KeyError as exc:  # expected
                msg = str(exc)
                has_hint = "Did you mean" in msg
                print(f"{attempt!r:30} hint={has_hint!s:5} message={msg}")


if __name__ == "__main__":
    main()
