"""Measure, per provider: the raise-line coverage delta and the mutation verdicts.

Every number the figure renders comes from this run; nothing is hand-typed.
"""
from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

import strands_robots

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)

TESTFILE = "tests/policies/test_state_key_name_list_contract.py"
NEW = [
    "test_the_behavioural_table_covers_every_surface_that_owns_the_check",
    "test_every_owning_provider_refuses_the_bare_string",
    "test_every_owning_provider_refuses_a_non_string_entry",
    "test_a_refusal_leaves_the_previously_bound_layout",
    "test_the_validate_only_provider_stores_nothing_either_way",
    "test_every_owning_provider_accepts_a_distinct_list",
]
NEW_K = " or ".join(NEW)
OLD_K = f"not ({NEW_K})"

PROVIDERS = [
    ("cosmos3", "strands_robots/policies/cosmos3/policy.py"),
    ("curobo", "strands_robots/policies/curobo/policy.py"),
    ("groot", "strands_robots/policies/groot/policy.py"),
    ("lerobot_async", "strands_robots/policies/lerobot_async/policy.py"),
    ("lerobot_local", "strands_robots/policies/lerobot_local/policy.py"),
    ("moveit2", "strands_robots/policies/moveit2/policy.py"),
    ("vera", "strands_robots/policies/vera/provider.py"),
]
RAISE = "            raise ValueError(error)\n"


def fn_range(src: str) -> tuple[int, int]:
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == "set_robot_state_keys":
            return n.lineno, n.end_lineno
    raise AssertionError("setter not found")


def raise_line(path: str) -> int:
    src = pathlib.Path(path).read_text()
    lo, hi = fn_range(src)
    lines = src.splitlines()
    for line in range(lo, hi + 1):
        if lines[line - 1].strip().startswith("raise ValueError"):
            return line
    raise AssertionError(f"no raise in {path}")


def cov(k: str, out: str) -> dict:
    subprocess.run(
        [sys.executable, "-m", "pytest", TESTFILE, "-q", "-p", "no:randomly", "-k", k,
         "--cov=strands_robots", f"--cov-report=json:{out}", "--cov-fail-under=0"],
        capture_output=True, text=True, check=False,
    )
    return json.load(open(out))


def run(k: str) -> tuple[int, int]:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTFILE, "-q", "--no-cov", "-p", "no:randomly", "-k", k],
        capture_output=True, text=True, check=False,
    )
    passed = failed = 0
    for line in r.stdout.splitlines():
        if " passed" in line or " failed" in line:
            for part in line.replace("=", " ").split(","):
                bits = part.split()
                for i, b in enumerate(bits):
                    if b == "passed" and i:
                        passed = int(bits[i - 1])
                    if b == "failed" and i:
                        failed = int(bits[i - 1])
    return passed, failed


facts: dict = {"tree": TREE, "providers": {}, "mutations": [], "counts": {}}

before = cov(OLD_K, "/tmp/art-before.json")
after = cov("", "/tmp/art-after.json")
facts["counts"]["before_tests"] = run(OLD_K)[0]
facts["counts"]["after_tests"] = run("")[0]

for label, path in PROVIDERS:
    line = raise_line(path)
    facts["providers"][label] = {
        "path": path,
        "line": line,
        "before": line in before["files"][path]["executed_lines"],
        "after": line in after["files"][path]["executed_lines"],
    }


def mutate(path: str, style: str) -> tuple[str, str]:
    src = pathlib.Path(path).read_text()
    lo, hi = fn_range(src)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo - 1 : hi])
    assert region.count(RAISE) == 1
    if style == "discard the raise, keep the call":
        new = region.replace(RAISE, "            pass  # MUTATION\n", 1)
    elif style == "re-word the message locally":
        new = region.replace(RAISE, '            raise ValueError("bad robot_state_keys")\n', 1)
    else:
        guard = (
            "        if robot_state_keys and (\n"
            '            error := name_list_error(robot_state_keys, "robot_state_keys", '
            '"set_robot_state_keys")\n        ):\n' + RAISE
        )
        assert region.count(guard) == 1, path
        new = region.replace(guard, "", 1)
    out = "".join(lines[: lo - 1]) + new + "".join(lines[hi:])
    ast.parse(out)
    return src, out


for style, targets in (
    ("discard the raise, keep the call", PROVIDERS),
    ("re-word the message locally", [PROVIDERS[0]]),
    ("delete the whole guard", [PROVIDERS[0]]),
):
    for label, path in targets:
        src, mutated = mutate(path, style)
        p = pathlib.Path(path)
        try:
            p.write_text(mutated)
            new_p, new_f = run(NEW_K)
            old_p, old_f = run(OLD_K)
        finally:
            p.write_text(src)
            assert p.read_text() == src, f"restore failed for {path}"
        facts["mutations"].append(
            {"style": style, "provider": label,
             "new_failed": new_f, "new_passed": new_p,
             "old_failed": old_f, "old_passed": old_p}
        )
        print(f"  {style:34s} {label:14s} new={new_f}F/{new_p}P  pre-existing={old_f}F/{old_p}P")

pathlib.Path("/tmp/art-facts.json").write_text(json.dumps(facts, indent=2))
print("counts:", facts["counts"])
print("WROTE /tmp/art-facts.json")
