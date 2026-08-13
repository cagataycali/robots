"""Mutate the SUCCESS path to substitute MockPolicy for every provider.

The three existing pins assert `import_policy_class("mock") is MockPolicy`,
which stays TRUE under this mutation - so if nothing else catches it, the
class docstring's own subject ("no provider is ever substituted") is unpinned
in the direction a raises assertion cannot reach.
"""
from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
import strands_robots
assert pathlib.Path(strands_robots.__file__).parents[1] == ROOT
print("TREE:", ROOT)

SRC = ROOT / "strands_robots" / "registry" / "policies.py"
original = SRC.read_text(encoding="utf-8")

OLD = '        return getattr(mod, config["class"])\n'
NEW = """        from strands_robots.policies.mock import MockPolicy

        return MockPolicy
"""

tree = ast.parse(original)
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "import_policy_class")
region = "\n".join(original.splitlines()[fn.lineno - 1: fn.end_lineno]) + "\n"
print(f"anchor: in_fn={region.count(OLD)} in_file={original.count(OLD)}")
assert region.count(OLD) == 1, "anchor not unique inside import_policy_class"


def run(paths: list[str], label: str) -> tuple[int, list[str]]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q", "--no-cov", "-p", "no:randomly", "--tb=no"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )
    failed = [l.split(" ")[0] for l in proc.stdout.splitlines() if l.startswith("FAILED")]
    tail = [l for l in proc.stdout.splitlines() if " passed" in l or " failed" in l]
    print(f"  [{label}] {tail[-1] if tail else 'no summary'}")
    return len(failed), failed


SCOPE = ["tests/registry", "tests/test_registry.py", "tests/policies/test_factory.py"]
facts: dict[str, object] = {"tree": str(ROOT)}
try:
    SRC.write_text(original.replace(OLD, NEW, 1), encoding="utf-8")
    assert SRC.read_text(encoding="utf-8") != original
    print("\nMUTATION: every provider's SUCCESS path returns MockPolicy")
    n, f = run(SCOPE, "registry + factory suites")
    facts["success_mutation_failed"] = f
finally:
    SRC.write_text(original, encoding="utf-8")
    assert SRC.read_text(encoding="utf-8") == original, "RESTORE FAILED"
    print("restored: byte-identical")

print(f"\ncaught by the existing suites: {len(facts['success_mutation_failed'])} failures")
for name in facts["success_mutation_failed"]:  # type: ignore[union-attr]
    print("   ", name)
json.dump(facts, open(f"/tmp/successmut-{sys.argv[1]}.json", "w"), indent=2)
