import ast, json, os, pathlib, re, subprocess, sys

run = os.environ["GITHUB_RUN_ID"]
SRC = pathlib.Path("strands_robots/utils.py")
NEW_K = ("TestTheVerdictHalfAnswersAnUnreadableLength or TestTheCallersThatCountByReadingAgain "
         "or test_an_acceptable_lazy_vector_is_still_accepted")
FILES = ["tests/test_unsized_value_is_refused_not_raised.py",
         "tests/test_container_refusals_render_elementwise.py"]

GATE = """    if sequence_length(vec) is None:
        # The components were read and were finite; what the value cannot supply
        # is a length for the caller to count. Same words as the sibling
        # coercions, because it is the same verdict about the same value.
        return f"{method}: '{param_name}' must be a list/tuple of numbers, got {_refusal_container_repr(vec)}"
"""

MUTATIONS = [
    ("M1 revert the fix (delete the probe)", GATE, ""),
    ("M2 probe BEFORE the component read",
     "    err = _read_finite_vector(method, param_name, vec)[1]\n    if err is not None:\n        return err\n" + GATE,
     GATE + "    err = _read_finite_vector(method, param_name, vec)[1]\n    if err is not None:\n        return err\n"),
    ("M3 reword the verdict locally",
     """        return f"{method}: '{param_name}' must be a list/tuple of numbers, got {_refusal_container_repr(vec)}"\n    return None""",
     """        return f"{method}: '{param_name}' must be a sized sequence of numbers"\n    return None"""),
    ("M4 own narrow len() probe instead of the shared owner",
     "    if sequence_length(vec) is None:",
     "    try:\n        len(vec)\n    except TypeError:\n        pass\n    else:\n        return None\n    if True:"),
    ("M5 keep the probe, discard its verdict",
     """        return f"{method}: '{param_name}' must be a list/tuple of numbers, got {_refusal_container_repr(vec)}"\n    return None""",
     """        pass\n    return None"""),
]

def fn_range(text: str, name: str) -> tuple[int, int]:
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return fn.lineno, fn.end_lineno or fn.lineno

def run_arm(k: str | None) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:randomly", "--no-cov", *FILES]
    if k: cmd += ["-k", k]
    out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"}).stdout
    m = re.search(r"(\d+) failed", out)
    return int(m.group(1)) if m else 0

original = SRC.read_text()
lo, hi = fn_range(original, "finite_vector_error")
region = "".join(original.splitlines(keepends=True)[lo - 1:hi])
rows = []
try:
    for label, old, new in MUTATIONS:
        in_fn, in_file = region.count(old), original.count(old)
        assert in_fn == 1, f"{label}: anchor in_fn={in_fn} in_file={in_file}"
        print(f"  [{label}] anchor in_fn={in_fn} in_file={in_file}")
        mutated = original.replace(old, new, 1)
        assert mutated != original
        ast.parse(mutated)
        SRC.write_text(mutated)
        new_f, old_f = run_arm(NEW_K), run_arm(f"not ({NEW_K})")
        rows.append({"label": label, "new_fail": new_f, "old_fail": old_f})
        print(f"    -> new cases: {new_f} failed | pre-existing: {old_f} failed")
        SRC.write_text(original)
    SRC.write_text(original)
    base_new, base_old = run_arm(NEW_K), run_arm(f"not ({NEW_K})")
    rows.append({"label": "unmutated control", "new_fail": base_new, "old_fail": base_old})
    print(f"  [control] new: {base_new} failed | pre-existing: {base_old} failed")
finally:
    SRC.write_text(original)
    assert SRC.read_text() == original

caught = sum(1 for r in rows if r["label"].startswith("M") and r["new_fail"] > 0)
blind = sum(1 for r in rows if r["label"].startswith("M") and r["old_fail"] == 0)
n_mut = sum(1 for r in rows if r["label"].startswith("M"))
print(f"\nCAUGHT by the new cases: {caught}/{n_mut}   BLIND to the pre-existing: {blind}/{n_mut}")
pathlib.Path(f"/tmp/mut-{run}.json").write_text(json.dumps({"rows": rows, "caught": caught, "blind": blind, "n": n_mut}, indent=2))
