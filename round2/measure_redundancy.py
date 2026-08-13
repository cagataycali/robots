"""Mutate the funnel to substitute `mock`, and see which tests catch it.

If the sibling `pytest.raises` tests already catch a substitution, the test
under repair - whose `except ImportError: pass` exists only so its
return-value assertion can run - adds nothing.
"""
from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
import strands_robots

print("TREE:", pathlib.Path(strands_robots.__file__).parents[1])
assert pathlib.Path(strands_robots.__file__).parents[1] == ROOT, "measuring the wrong tree"

SRC = ROOT / "strands_robots" / "registry" / "policies.py"
TEST = "tests/registry/test_provider_import_error_names_its_remedy.py"
REPAIR = "test_the_funnel_never_falls_back_to_the_mock_provider"

original = SRC.read_text(encoding="utf-8")

# The failure path in the config branch: raise -> substitute the mock class.
OLD = """        except ImportError as exc:
            # A provider whose module needs an optional dependency at import
            # time (lerobot_local imports torch) otherwise raises a bare
            # "No module named 'torch'" naming neither this provider nor the
            # remedy - the dead end _provider_import_error exists to close.
            raise _provider_import_error(canonical, exc, config.get("extra")) from exc
"""
NEW = """        except ImportError:
            from strands_robots.policies.mock import MockPolicy

            return MockPolicy
"""
# Scope the anchor to import_policy_class and prove it is unique there.
tree = ast.parse(original)
fn = next(
    n for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "import_policy_class"
)
region = "\n".join(original.splitlines()[fn.lineno - 1: fn.end_lineno]) + "\n"
in_fn, in_file = region.count(OLD), original.count(OLD)
print(f"anchor: in_fn={in_fn} in_file={in_file}")
assert in_fn == 1 and in_file == 1, "anchor is not unique"


def run(node_ids: list[str], label: str) -> tuple[int, list[str]]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *node_ids, "-q", "--no-cov", "-p", "no:randomly", "--tb=no"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    failed = [
        l.split("::")[-1].split(" ")[0]
        for l in proc.stdout.splitlines()
        if l.startswith("FAILED")
    ]
    tail = [l for l in proc.stdout.splitlines() if " passed" in l or " failed" in l]
    print(f"  [{label}] {tail[-1] if tail else 'no summary'}")
    return len(failed), failed


facts: dict[str, object] = {"tree": str(ROOT)}
try:
    SRC.write_text(original.replace(OLD, NEW), encoding="utf-8")
    assert SRC.read_text(encoding="utf-8") != original, "mutation did not apply"

    print("\nMUTATION: the funnel substitutes MockPolicy instead of reporting")
    n_all, f_all = run([TEST], "whole module")
    facts["mutation_failed_whole_module"] = f_all

    n_sib, _ = run([f"{TEST}", "-k", f"not {REPAIR}"], "siblings only (the repaired test deselected)")
    facts["mutation_failed_siblings_only"] = n_sib

    n_rep, _ = run([f"{TEST}::TestNoProviderIsSubstituted::{REPAIR}"], "the repaired test alone")
    facts["mutation_failed_repaired_alone"] = n_rep
finally:
    SRC.write_text(original, encoding="utf-8")
    assert SRC.read_text(encoding="utf-8") == original, "RESTORE FAILED"
    print("\nrestored:", "byte-identical" if SRC.read_text(encoding="utf-8") == original else "MISMATCH")

print("\n=== verdict ===")
print(f"substitution caught by siblings alone: {facts['mutation_failed_siblings_only']} failures")
print(f"substitution caught by the repaired test alone: {facts['mutation_failed_repaired_alone']} failures")
print("failing tests (whole module):")
for name in facts["mutation_failed_whole_module"]:  # type: ignore[union-attr]
    print("   ", name)
json.dump(facts, open(f"/tmp/redundancy-{sys.argv[1]}.json", "w"), indent=2)
