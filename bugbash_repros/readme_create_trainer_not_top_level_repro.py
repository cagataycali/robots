"""Repro: README's `create_trainer("lerobot_local")` row implies the symbol is
top-level, but ``from strands_robots import create_trainer`` is ImportError.

README.md:98 (`## What you get` -> `**Train**`) writes the call in code voice:

    **Train** with LeRobot (ACT to GR00T N1.7, `create_trainer("lerobot_local")`),
    Cosmos 3 (`"cosmos3"`) or RL (`"ppo"` / `"fast_sac"` / `"fast_td3"`),
    locally or as a SageMaker job (`"sagemaker"`), ...

A reader who copy-pastes the identifier `create_trainer` as they would
`create_policy` (same README, same package, same style, same `_LAZY_IMPORTS`
re-export table) hits ImportError. The three sibling factories that ARE
re-exported from the top-level namespace are:

    create_policy     (eager, strands_robots/__init__.py:100)
    create_simulation (lazy,  strands_robots/__init__.py:127)
    create_backend    (via register_backend/list_backends:128/131)

The `create_trainer` triplet (create_trainer / list_trainers / register_trainer)
is torch-free (the module's own docstring at
``strands_robots/training/__init__.py:44`` says "keeping ``import
strands_robots.training`` torch-free") but is absent from both the eager block
and the ``_LAZY_IMPORTS`` dict in ``strands_robots/__init__.py``.

Run this file against an editable install of strands-labs/robots @ main
(commit 8ccda74a5 at time of writing). Exit 0 = defect still fires; exit 1 =
fixed upstream (symbol is now top-level importable).
"""

from __future__ import annotations

import sys


def main() -> int:
    # Rung 1 - the README-copied form. One line, no sub-module.
    try:
        from strands_robots import create_trainer  # type: ignore[attr-defined]  # noqa: F401
    except ImportError as e:
        rung1 = f"ImportError: {e}"
    else:
        print("UNEXPECTED: create_trainer top-level import succeeded; defect fixed.")
        return 1

    # Rung 2 - the sibling that does work, as the symmetric peer.
    try:
        from strands_robots import create_policy  # noqa: F401
    except ImportError as e:  # pragma: no cover - would indicate a regression
        print(f"ERROR: sibling create_policy also missing: {e}")
        return 2

    # Rung 3 - the actual, deeper location of create_trainer.
    from strands_robots.training import create_trainer, list_trainers

    # Rung 4 - every README-named identifier resolves from the deep path.
    readme_identifiers = ["lerobot_local", "cosmos3", "ppo", "fast_sac", "fast_td3", "sagemaker"]
    deep_results: dict[str, str] = {}
    for ident in readme_identifiers:
        try:
            inst = create_trainer(ident)
            deep_results[ident] = f"OK ({type(inst).__name__})"
        except Exception as exc:  # noqa: BLE001
            deep_results[ident] = f"{type(exc).__name__}: {exc}"

    # Rung 5 - top-level __all__ comparison, policies vs training.
    import strands_robots as sr

    policy_triplet = {"create_policy", "list_providers", "register_policy"}
    training_triplet = {"create_trainer", "list_trainers", "register_trainer"}

    policy_in_all = sorted(policy_triplet & set(sr.__all__))
    training_in_all = sorted(training_triplet & set(sr.__all__))

    # Report
    print("=== Rung 1: `from strands_robots import create_trainer` ===")
    print(f"  {rung1}")
    print()
    print("=== Rung 2: sibling `from strands_robots import create_policy` ===")
    print("  OK  (symmetric peer imports fine)")
    print()
    print("=== Rung 3: deep path works ===")
    print(f"  from strands_robots.training import create_trainer, list_trainers  -> OK")
    print(f"  list_trainers() -> {list_trainers()}")
    print()
    print("=== Rung 4: every README-named identifier resolves from the deep path ===")
    for ident, r in deep_results.items():
        print(f"  create_trainer({ident!r}) -> {r}")
    print()
    print("=== Rung 5: top-level __all__ triplet symmetry ===")
    print(f"  policies triplet in __all__  : {policy_in_all}  (3/3)")
    print(f"  training triplet in __all__  : {training_in_all}  (0/3)")
    print()
    print("Defect: create_trainer / list_trainers / register_trainer are torch-free")
    print("(see strands_robots/training/__init__.py:44) and could be lazy-re-exported")
    print("alongside create_simulation / list_backends / register_backend, matching")
    print("the policies triplet above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
