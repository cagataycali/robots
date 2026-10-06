"""
Repro: `create_policy("microduck")` and `sim.run_policy(policy_provider="microduck")`
without `onnx_path` get a constructor-level ValueError that mentions `session`
(test-only), instead of the factory-level `policy_requires_error` refusal that
sibling ONNX provider `rsl_rl_onnx` gets.

Root cause: `strands_robots/registry/policies.json` declares `microduck.requires = []`
even though `MicroduckPolicy.__init__` enforces onnx_path (or session, test-only) at
runtime. `_required_keywords` reads the constructor signature which also sees
`onnx_path: str | Path | None = None`, so it finds nothing. The registry's
`requires` is the oracle `policy_requires_error` uses (strands_robots/registry/
policies.py:255-271) to emit the crisp "builds its policy from onnx_path, and none
was given. Pass onnx_path=... Without it ..." message.

Fix on this branch: declare `requires: ["onnx_path"]` for microduck + add an
`onnx_path` entry to the hints dict in `policy_requires_error`.

Run: python bugbash_repros/microduck_missing_onnx_path_repro.py
"""
from __future__ import annotations

import os
import sys
import traceback


def _try_create_policy(provider: str) -> None:
    """Call create_policy(provider) with no kwargs and print the error we got."""
    from strands_robots.policies import create_policy

    try:
        create_policy(provider)
        print(f"  [{provider}] UNEXPECTED SUCCESS (expected ValueError)")
    except Exception as exc:  # noqa: BLE001 - we are reporting the UX
        print(f"  [{provider}] {type(exc).__name__}: {exc}")


def _try_run_policy(provider: str) -> None:
    """Call sim.run_policy(policy_provider=provider) and print status + text."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from strands_robots.simulation import create_simulation

    sim = create_simulation("mujoco")
    sim.create_world()
    # microduck as the body - any body with a 14-DOF arm would do, the refusal
    # is pre-rollout.
    sim.add_robot("microduck")
    try:
        r = sim.run_policy(
            robot_name="microduck",
            policy_provider=provider,
            duration=0.1,
            control_frequency=50.0,
        )
        print(f"  [{provider}] status={r.get('status')!r}")
        print(f"  [{provider}] text={r['content'][0]['text'][:300]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [{provider}] {type(exc).__name__}: {exc}")


def main() -> int:
    print("== Repro: microduck factory-level refusal is missing ==")
    print()
    print("--- create_policy(<provider>) with NO kwargs ---")
    _try_create_policy("microduck")
    _try_create_policy("protomotions")  # same gap - identical shape
    _try_create_policy("rsl_rl_onnx")  # the one that gets it right

    print()
    print("--- sim.run_policy(policy_provider=<provider>) with NO policy_config ---")
    try:
        _try_run_policy("microduck")
        _try_run_policy("rsl_rl_onnx")
    except Exception:  # noqa: BLE001 - keep going
        traceback.print_exc()

    print()
    print("--- registry introspection ---")
    import json
    import pathlib

    reg_path = pathlib.Path("strands_robots/registry/policies.json")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    for name in ("microduck", "protomotions", "rsl_rl_onnx"):
        entry = reg["providers"][name]
        print(f"  {name:12s} requires={entry.get('requires')!r} config_keys has onnx_path="
              f"{'onnx_path' in entry.get('config_keys', [])}")

    print()
    print("--- what the factory oracle would say if `requires` listed onnx_path ---")
    from strands_robots.registry.policies import policy_requires_error

    for name in ("microduck", "protomotions", "rsl_rl_onnx"):
        msg = policy_requires_error(name, {}, "create_policy", "it cannot download or load any weight")
        print(f"  [{name}] -> {msg!r}")

    print()
    print("Observe: on main, microduck+protomotions get the thin constructor error that "
          "names `session` (test-only) alongside `onnx_path` and offers no shipped-weights hint; "
          "rsl_rl_onnx gets the crisp factory refusal naming where to get its onnx_path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
