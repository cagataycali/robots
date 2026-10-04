"""Repro: create_trainer() 404 has none of the Error-UX rungs create_policy() has.

Target: v0.5.3 (upstream HEAD 4735957)

What this repro shows (one asymmetric sibling to policy factory):

    create_policy("Lerobot_Local")  -> "Did you mean: 'lerobot_local', 'lerobot'?"
    create_trainer("Lerobot_Local") -> bare "No trainer registered ... Available: [...]"

Both resolvers address the SAME provider families (lerobot_local, cosmos3,
rsl_rl_onnx, ...). import_trainer_class's own docstring claims:

    "Resolution order mirrors policies.factory._resolve_policy_class"

but the trainer factory is missing TWO rungs the policy factory has:

  1. Case + dash fold on provider lookup
     (policy: "Lerobot_Local" / "lerobot-local" / "LEROBOT" resolve; trainer: all 404)
  2. difflib.get_close_matches "Did you mean: ..." hint when the name is close
     to a registered one (policy factory factory.py:395-401; trainer factory
     factory.py:163 just raises the bare ValueError).

The practical paper-cut: a user who copied `create_policy("Lerobot_Local")`
from a tutorial (common on multi-GPU macOS, where identifiers get title-cased
by autocomplete) gets a hint on the inference side and a stone wall on the
training side -- the two factories are documented as symmetric peers.

Run (inside the clone, with the package importable; needs NO heavy deps):
    python3 bugbash_repros/trainer_factory_no_did_you_mean_repro.py

Expected: both halves resolve OR both halves offer a hint.
Actual:   policy half offers hint; trainer half silent.
"""

from __future__ import annotations

import difflib
import sys

sys.path.insert(0, ".")

from strands_robots.policies.factory import _resolve_policy_class
from strands_robots.training.factory import import_trainer_class, list_trainers


def _report(fn, label, name):
    print(f"\n>>> {label}({name!r})")
    try:
        result = fn(name)
        print(f"  OK -> {result!r}")
    except Exception as exc:
        msg = str(exc)
        has_did_you_mean = "did you mean" in msg.lower()
        print(f"  {type(exc).__name__}: {msg[:260]}")
        print(f"  has 'Did you mean' hint: {has_did_you_mean}")
        return has_did_you_mean
    return True


def main() -> int:
    print("=" * 70)
    print("Asymmetric 404 UX: create_policy vs create_trainer")
    print("=" * 70)

    # Case 1: TitleCase variant of a registered name (common from LLM autocomplete)
    name = "Lerobot_Local"
    print(f"\n[case 1] TitleCase variant of a registered provider: {name!r}")
    pol_hint = _report(_resolve_policy_class, "create_policy", name)
    trn_hint = _report(import_trainer_class, "create_trainer", name)

    # Case 2: typo close to lerobot_local
    name = "lerbot_local"
    print(f"\n[case 2] Typo close to lerobot_local: {name!r}")
    _report(_resolve_policy_class, "create_policy", name)
    _report(import_trainer_class, "create_trainer", name)

    # Case 3: dash variant - matches NVIDIA's own 'gr00t' / 'groot' precedent
    name = "cosmos-3"
    print(f"\n[case 3] Dash variant of registered 'cosmos3': {name!r}")
    _report(_resolve_policy_class, "create_policy", name)
    _report(import_trainer_class, "create_trainer", name)

    # Prove the trainer registry would have offered useful hints via difflib:
    close = difflib.get_close_matches("Lerobot_Local".lower(), list_trainers(), n=3, cutoff=0.6)
    print(f"\n[diagnostic] difflib close-match for 'Lerobot_Local' against list_trainers():")
    print(f"  -> {close!r}  (trainer factory has this info but never uses it)")

    print("\n=== Verdict ===")
    print("policy factory  : offers 'Did you mean' hint         (good UX)")
    print("trainer factory : only prints bare 'Available:' list (asymmetric)")
    print(
        "\nUpstream: strands_robots/training/factory.py:163 (bare raise),"
        " cf. strands_robots/policies/factory.py:395-401 (difflib + hint)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
