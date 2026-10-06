"""Repro: 8 policy providers in strands_robots/registry/policies.json omit the
``extra`` field, so an ImportError on their optional dependency produces the
GENERIC "Install 'X' (or the strands-robots extra that ships it)" message
with no actionable ``pip install 'strands-robots[<extra>]'`` line.

Sibling providers in the same JSON (lerobot_local, wbc_latent, rsl_rl_onnx)
DO carry the field and so DO surface the install command. The 8 affected
providers (wbc, kimodo, microduck, protomotions, cosmos3, moveit2,
holosoma, rl) all have real deps in pyproject.toml's
``[project.optional-dependencies]`` with matching names, so the fix is a
mechanical "add ``extra`` to the JSON entry". Pure JSON edit, no code change.

Run with no args: BEFORE/AFTER-neutral - checks declaration + message shape.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from strands_robots.policies.factory import _provider_import_error
from strands_robots.registry.policies import get_policy_provider

# Providers that have a matching pyproject.toml extra by canonical name:
AFFECTED = {
    "wbc": "wbc",
    "kimodo": "kimodo",
    "microduck": "microduck",
    "protomotions": "protomotions",
    "cosmos3": "cosmos3-service",  # default backend per cosmos3/__init__.py:18
    "moveit2": "moveit2",
    "holosoma": "holosoma",
    "rl": "rl",
}

# Controls that already declare their extra correctly:
CONTROLS = {
    "lerobot_local": "lerobot",
    "wbc_latent": "wbc",
    "rsl_rl_onnx": "sim-mjlab",
}


def message_user_sees(provider_name: str) -> str:
    """Format the error the factory would raise if provider's dep were missing."""
    cfg = get_policy_provider(provider_name)
    exc = ImportError("No module named 'onnxruntime'")
    exc.name = "onnxruntime"
    err = _provider_import_error(provider_name, exc, cfg.get("extra"))
    return str(err)


def check_declared(provider: str, expected_extra: str) -> tuple[bool, str]:
    cfg = get_policy_provider(provider)
    actual = cfg.get("extra")
    if actual == expected_extra:
        return True, f"OK (extra={actual!r})"
    return False, f"missing/wrong (expected {expected_extra!r}, got {actual!r})"


def check_message_actionable(provider: str) -> tuple[bool, str]:
    msg = message_user_sees(provider)
    if "pip install 'strands-robots[" in msg:
        return True, "actionable install hint present"
    return False, "GENERIC dead-end wording"


print("=" * 72)
print("AFFECTED providers (should declare an extra matching pyproject.toml):")
print("=" * 72)
affected_ok = 0
for prov, expected in AFFECTED.items():
    ok1, m1 = check_declared(prov, expected)
    ok2, m2 = check_message_actionable(prov)
    status = "PASS" if (ok1 and ok2) else "FAIL"
    print(f"  [{status}] {prov}: declared={m1}; message={m2}")
    affected_ok += int(ok1 and ok2)

print()
print("=" * 72)
print("CONTROL providers (declare correctly today; regression guard):")
print("=" * 72)
control_ok = 0
for prov, expected in CONTROLS.items():
    ok1, m1 = check_declared(prov, expected)
    ok2, m2 = check_message_actionable(prov)
    status = "PASS" if (ok1 and ok2) else "FAIL"
    print(f"  [{status}] {prov}: declared={m1}; message={m2}")
    control_ok += int(ok1 and ok2)

print()
print(f"AFFECTED: {affected_ok}/{len(AFFECTED)} pass")
print(f"CONTROL:  {control_ok}/{len(CONTROLS)} pass")

if affected_ok == len(AFFECTED) and control_ok == len(CONTROLS):
    print("\nALL PASS - fix is live. Close harness issue.")
    sys.exit(0)
else:
    print("\nFAIL - defect reproduces.")
    print()
    print("Sample user-facing message BEFORE the fix:")
    print("-" * 72)
    # Show a representative failing message
    for prov in AFFECTED:
        cfg = get_policy_provider(prov)
        if cfg.get("extra") is None:
            print(f"  create_policy({prov!r}) with its dep missing says:")
            print()
            for line in message_user_sees(prov).splitlines():
                print(f"    {line}")
            print()
            print(f"  (fix: add `\"extra\": \"{AFFECTED[prov]}\"` to the")
            print(f"  providers[{prov!r}] entry in")
            print(f"  strands_robots/registry/policies.json, mirroring")
            print(f"  wbc_latent which already declares extra='wbc'.)")
            break
    sys.exit(1)
