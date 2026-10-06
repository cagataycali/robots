"""Repro: pyproject.toml `[all]` extra silently omits 5 policy providers.

A user who follows the Python idiom `pip install 'strands-robots[all]'`
expecting a batteries-included install will NOT be able to run these
`create_policy(...)` providers:

    * microduck      (needs strands-robots[microduck])
    * cosmos3        (needs strands-robots[cosmos3-service] / [cosmos3-sim])
    * curobo         (needs strands-robots[curobo])
    * flux3_action   (needs strands-robots[flux3])
    * rsl_rl_onnx    (needs strands-robots[sim-mjlab])

All five are:
    1. Advertised in `strands_robots/registry/policies.json` as supported
       `policy_provider` values.
    2. Backed by a *dedicated* `pyproject.toml` extras group that declares
       the exact missing deps.
    3. **Not** cited anywhere in `pyproject.toml`'s `[all]` list, even
       though 10 sibling policy extras ARE cited there
       (`[wbc]`, `[kimodo]`, `[protomotions]`, `[holosoma]`, `[moveit2]`,
       `[lerobot]`, `[rl]`, `[sim-mujoco]`, `[sim-urdf]`, `[mesh]`).

Only `voice` has a comment explaining its omission (awscrt pin clash);
the five below have NO comment and no documented reason.

Static-truth check — no package install needed.

Upstream file:line:
    pyproject.toml:815-844 (the `all = [...]` list)
    strands_robots/registry/policies.json (16 providers advertised)

Run:
    python bugbash_repros/all_extra_omits_5_policy_providers_repro.py
Exit code 1 on the defect.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Load pyproject.toml extras.
    with (repo_root / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    extras = pyproject["project"]["optional-dependencies"]

    # 2. Load policy providers that are advertised in the registry.
    with (repo_root / "strands_robots" / "registry" / "policies.json").open() as f:
        provs = json.load(f)["providers"]

    # 3. Compute what [all] cites.
    all_deps = extras.get("all", [])
    in_all: set[str] = set()
    for dep in all_deps:
        m = re.search(r"strands-robots\[([^\]]+)\]", dep)
        if m:
            in_all.add(m.group(1))

    # 4. Mapping: provider -> dedicated pyproject extra group.
    #    (The ones whose module needs an optional dep AND whose extra exists.)
    #    policies.json has `extra=None` for most entries today (that's a
    #    separate defect, cagataycali/robots-harness#841), so we read the
    #    extras from pyproject by convention/naming.
    provider_to_extra = {
        "microduck": "microduck",
        "cosmos3": "cosmos3-service",
        "curobo": "curobo",
        "flux3_action": "flux3",
        "rsl_rl_onnx": "sim-mjlab",
    }

    missing: list[tuple[str, str]] = []
    for provider, extra in provider_to_extra.items():
        assert provider in provs, f"provider {provider!r} not in policies.json"
        assert extra in extras, f"extra [{extra}] not in pyproject"
        if extra not in in_all:
            missing.append((provider, extra))

    # 5. Also report the 10 siblings that ARE in [all] (for symmetry).
    sibling_providers_in_all = {
        "lerobot_local": "lerobot",
        "moveit2": "moveit2",
        "wbc": "wbc",
        "wbc_gait": "wbc",
        "wbc_latent": "wbc",
        "holosoma": "holosoma",
        "kimodo": "kimodo",
        "protomotions": "protomotions",
        "rl": "rl",
    }
    present = [
        (p, e) for p, e in sibling_providers_in_all.items() if e in in_all
    ]

    # 6. Report.
    print(f"[all] cites {len(in_all)} extras: {sorted(in_all)}\n")
    print(f"Policy providers whose extra IS in [all] ({len(present)}):")
    for p, e in sorted(present):
        print(f"    create_policy({p!r:22s}) <- strands-robots[{e}]")
    print()
    print(
        f"🔴 Policy providers whose extra is MISSING from [all] "
        f"({len(missing)}):"
    )
    for p, e in missing:
        print(f"    create_policy({p!r:22s}) <- strands-robots[{e}]   (NOT in [all])")

    print()
    if missing:
        print(
            "DEFECT: `pip install 'strands-robots[all]'` leaves these providers "
            "with unimportable modules even though pyproject HAS a dedicated "
            "extras group that would fix each one."
        )
        print(
            "Only [voice] has a comment explaining its omission from [all] "
            "(awscrt pin clash); these five have no comment and no documented "
            "reason."
        )
        return 1
    print("OK: [all] bundles every policy-provider extra.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
