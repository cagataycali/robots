"""Repro: README L93 'Any policy' bullet names `whole-body control` and `scripted`
and `GR00T N1.7`, but all three stone-wall `create_policy`:

- `whole-body control`: no alias to `wbc` and no entry in REMOVED_PROVIDERS;
  difflib's `get_close_matches` cutoff (0.4) returns no suggestions for the
  5-word phrase against the 16 1-word providers, so the error prints
  `Available: [...]` with NO 'Did you mean' hint, even though 'wbc' is right
  there in the list.
- `scripted`: NOT a provider, NOT a registered type, NOT in REMOVED_PROVIDERS.
  It is only an architectural category named in 3 docstrings in
  `strands_robots/policies/{base.py,__init__.py}`. The README's bullet reads
  it as a parallel item to the concrete providers around it. User trying
  `create_policy('scripted')` lands in the generic 'Unknown provider' refusal
  that lists 16 providers, none of which are a scripted trajectory player.
- `GR00T N1.7`: README's exact spelling (NVIDIA brand + version) does not
  match the lowercase 'groot' key in REMOVED_PROVIDERS. The redirect that
  would have told the user to use `lerobot_local(policy_type='groot')` is
  consequently NOT emitted.

Sibling providers in the SAME README bullet (`Cosmos 3`, `cuRobo`, `MoveIt2`)
all get a 'Did you mean' hint pointing at `cosmos3`, `curobo`, `moveit2`
through the shared difflib pipeline, because their canonical name is a short
single-token typo away. The failures above are a shape asymmetry, not a
difflib floor: adding aliases/REMOVED_PROVIDERS entries makes `wbc` and
`scripted` reachable with the same code path that cosmos3/curobo/moveit2
already use.

Distinct from harness#833 which covers only the LeRobot policy-type half of
the same bullet (`act`, `pi0`, `smolvla`, `diffusion`, `molmoact2`) --- those
have REMOVED_PROVIDERS entries that redirect correctly to lerobot_local. This
repro targets the NON-LeRobot half of the enumeration (`whole-body control`,
`scripted`, `GR00T N1.7`), which has NO redirect.

Upstream (strands-labs/robots@afc74e5):
- README.md:93 (the 'Any policy' bullet)
- strands_robots/registry/policies.py:77-91 (REMOVED_PROVIDERS, no wbc/scripted/`gr00t n1.7`)
- strands_robots/policies/factory.py:329 (generic 404 raised here)
- strands_robots/policies/factory.py:555 (difflib pipeline exists for kwargs)

Expected: either
  (a) README is rewritten to name only resolvable canonical spellings, or
  (b) REMOVED_PROVIDERS gains redirects for 'wbc' aliases ('whole-body control',
      'whole_body_control', 'whole-body-control') and `GR00T N1.7` canonical
      capitalisation + versioned forms, and `scripted` either becomes a
      resolvable path (CompositePolicy on top of a scripted trajectory) OR is
      struck from the README bullet.

Observed: `create_policy('whole-body control')` and `create_policy('scripted')`
raise ValueError 'Unknown policy provider' with NO 'Did you mean' hint;
`create_policy('GR00T N1.7')` raises the generic 404 instead of the groot
redirect.

Run (with strands-robots installed):
    python bugbash_repros/create_policy_readme_scripted_whole_body_stone_wall_repro.py
"""
from __future__ import annotations

import difflib
import sys

from strands_robots.policies import create_policy, list_providers
from strands_robots.registry.policies import REMOVED_PROVIDERS


def _try(name: str) -> tuple[bool, str, bool]:
    """Return (success, message, has_did_you_mean_hint)."""
    try:
        create_policy(name)
        return (True, "<success>", False)
    except Exception as e:
        msg = str(e)
        has_hint = "did you mean" in msg.lower()
        return (False, msg, has_hint)


def main() -> int:
    providers = list_providers()
    print(f"Registered providers ({len(providers)}): {providers}")
    print(f"REMOVED_PROVIDERS keys: {sorted(REMOVED_PROVIDERS.keys())}")
    print()

    # The exact strings README L93 names, as copy-paste from the README:
    # "LeRobot (ACT / Pi0 / SmolVLA / Diffusion / GR00T N1.7), Cosmos 3,
    #  MolmoAct2, whole-body control, cuRobo, MoveIt2, scripted"
    readme_names = {
        # The three NEW stone-walls (distinct from harness#833):
        "whole-body control": "no alias to 'wbc'",
        "scripted": "no provider at all; architectural category only",
        "GR00T N1.7": "NVIDIA brand + version; REMOVED_PROVIDERS key is lowercase 'groot'",
        # Three SIBLINGS that get a helpful hint, for contrast:
        "Cosmos 3": "has 'Did you mean cosmos3' hint via difflib",
        "cuRobo": "has 'Did you mean curobo' hint via difflib",
        "MoveIt2": "has 'Did you mean moveit2' hint via difflib",
    }

    stone_walls: list[str] = []
    good_hints: list[str] = []

    for name, note in readme_names.items():
        ok, msg, has_hint = _try(name)
        bucket = "WORKS" if ok else ("HAS HINT" if has_hint else "STONE WALL")
        print(f"[{bucket}] create_policy({name!r})")
        print(f"    note: {note}")
        print(f"    msg : {msg[:260]}")
        print()
        if not ok and not has_hint:
            stone_walls.append(name)
        elif has_hint:
            good_hints.append(name)

    # Prove difflib COULD have helped 'whole-body control' with a lower cutoff
    # or token-level matching. Current cutoff is 0.4 (shared `close_match_hint`).
    best = difflib.get_close_matches(
        "whole-body control", providers, n=3, cutoff=0.0
    )
    print(f"difflib(whole-body control, cutoff=0.0) best 3 of {len(providers)}: {best}")
    print(
        "   -> at cutoff=0.0 'wbc' DOES rank, but at the shared 0.4 cutoff it is\n"
        "      filtered out because 'whole-body control' / 'wbc' share no common\n"
        "      token. The fix is NOT a cutoff drop (false positives); it is an\n"
        "      alias entry on the WBC provider or a REMOVED_PROVIDERS redirect\n"
        "      matching the three dash/space/underscore spellings the README uses."
    )
    print()

    print(f"STONE WALLS (no hint, no redirect): {stone_walls}")
    print(f"GOOD   hints (siblings in SAME bullet): {good_hints}")

    # Assert the shape: >=3 stone walls, >=3 siblings with hints
    assert len(stone_walls) >= 3, f"Expected >=3 stone walls, got {stone_walls}"
    assert len(good_hints) >= 3, f"Expected >=3 siblings with hints, got {good_hints}"

    return 0


if __name__ == "__main__":
    sys.exit(main())
