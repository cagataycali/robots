"""Repro: Flux3ActionPolicy's `torch` require_optional preempts the policy's own
FLUX3_SYSTEM_INSTALL_HINT, nudging the user at the wrong extra.

Defect class: Error UX (bad order; unreachable useful message)
Target: v0.5.3, rotation `extras_gating`

Symptom
-------
On a fresh `pip install strands-robots` (no `[lerobot]`, no torch), constructing
`Flux3ActionPolicy()` raises an ImportError whose message says::

    'torch' is required for FLUX 3 Action inference (CUDA)
    Install with:
      pip install 'strands-robots[lerobot]'
      pip install torch

The policy module *already defines* `FLUX3_SYSTEM_INSTALL_HINT`, a NATTEN-aware
remedy that names the exact wheel pin and the fact that `flux-action` is
git-only. The second `require_optional("flux_action", system_install=...)` call
uses that hint correctly - but the user never sees it, because torch is
checked first with the wrong extra.

Why `extra="lerobot"` is actively misleading for flux3
------------------------------------------------------
- `[lerobot]` caps torch<2.12 (pyproject.toml:368, "lerobot 0.6 caps torch<2.12").
- flux3 docs mandate torch>=2.10 **paired with a NATTEN wheel** off
  https://whl.natten.org/ for that torch/CUDA build.
- Following the error (plain `pip install torch`) installs CPU-ish torch with
  no NATTEN wheel pair, so the policy's next construction step still fails -
  and the user is now one typo away from a mixed torch/lerobot pin fight on
  the Jetson Thor sm_110 bug the pyproject comment at line 368 cites.
- A sibling policy (`moveit2`, `kimodo`, `wbc`) correctly points at its OWN
  extra, not someone else's.
- Line 198-201 of the same file already uses FLUX3_SYSTEM_INSTALL_HINT.

Compare the shape to harness#864 (SpotDriver.connect_eagerly checks creds
before SDK import, so the useful '[spot] extra' message is unreachable
until creds are set). Same shape: a useful install hint shipped in the
same file is unreachable because an earlier guard raises with a wrong
nudge.

Upstream cite
-------------
strands_robots/policies/flux3_action/policy.py:197 (bad extra=)
strands_robots/policies/flux3_action/policy.py:49-56 (FLUX3_SYSTEM_INSTALL_HINT)
strands_robots/policies/flux3_action/policy.py:198-201 (correct use, line below)
docs/learn/policies/flux3-action.md:9-13 (authoritative install sequence)
pyproject.toml:184-192 ([flux3] = [] by design)
pyproject.toml:355-370 ([lerobot] torch cap <2.12)

Minimal failing example
-----------------------
This script hides `torch` from the import system, then constructs the
policy. It prints the raised ImportError, so the reader can see:

  * the error names `pip install 'strands-robots[lerobot]'`, not `[flux3]`.
  * the FLUX3_SYSTEM_INSTALL_HINT (git-only, NATTEN wheel) is never printed.
  * the `__context__` is `ModuleNotFoundError('torch')`.

Run with: `python flux3_torch_unreachable_system_hint_repro.py`
Expected exit: 2 (defect fires).
"""

from __future__ import annotations

import sys


class _HideTorch:
    """Meta-path finder that pretends `torch` and `torch.*` are not installed."""

    def find_spec(self, name, path, target=None):  # noqa: D401
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"No module named {name!r}")
        return None


def main() -> int:
    # Drop any already-imported torch and shadow future imports.
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            del sys.modules[mod]
    sys.meta_path.insert(0, _HideTorch())

    try:
        from strands_robots.policies.flux3_action.policy import (  # noqa: E402
            FLUX3_SYSTEM_INSTALL_HINT,
            Flux3ActionPolicy,
        )
    except ImportError as exc:
        print(f"UNEXPECTED import failure before construction: {exc}")
        return 1

    try:
        Flux3ActionPolicy()
    except ImportError as exc:
        msg = str(exc)
        print("ImportError raised on Flux3ActionPolicy() with torch hidden:")
        print("-" * 60)
        print(msg)
        print("-" * 60)
        print(f"exc.name = {exc.name!r}  (torch, as expected)")
        print(f"__context__ = {exc.__context__!r}")
        print()

        # Assertions - the defect fires if all three hold.
        wrong_extra_cited = "strands-robots[lerobot]" in msg
        own_extra_absent = "strands-robots[flux3]" not in msg
        system_hint_absent = "flux-action is not published on PyPI" not in msg

        print(f"wrong extra ([lerobot]) is in message:   {wrong_extra_cited}")
        print(f"own extra ([flux3]) is NOT in message:   {own_extra_absent}")
        print(f"FLUX3_SYSTEM_INSTALL_HINT not reachable: {system_hint_absent}")
        print()
        print("FLUX3_SYSTEM_INSTALL_HINT the user never sees:")
        print("-" * 60)
        print(FLUX3_SYSTEM_INSTALL_HINT)
        print("-" * 60)

        if wrong_extra_cited and own_extra_absent and system_hint_absent:
            print("\nDEFECT FIRES: torch hint names the wrong extra and masks the policy's own system_install hint.")
            return 2
        print("\nDEFECT DID NOT FIRE (one or more assertions failed).")
        return 0

    print("No ImportError raised - torch was not actually hidden?")
    return 1


if __name__ == "__main__":
    sys.exit(main())
