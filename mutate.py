"""Mutate the publication gate; measure the new module vs the pre-existing suite."""
import ast, pathlib, re, subprocess, sys

ROOT = pathlib.Path(sys.argv[1])
SRC = ROOT / "strands_robots/tools/lerobot_train.py"
NEW = "tests/tools/test_train_publication_requires_approval.py"
OLD = "tests/tools/test_lerobot_train.py"

GATE = """
            if push_to_hub:
                gate_err = _gate_extra_flags({"policy.push_to_hub": push_to_hub}, tool_context)
                if gate_err:
                    return gate_err
"""

MUTATIONS = [
    ("M1 delete the push_to_hub gate", "lerobot_train", GATE, "\n"),
    ("M2 keep the call, discard the refusal", "lerobot_train",
     '                gate_err = _gate_extra_flags({"policy.push_to_hub": push_to_hub}, tool_context)\n'
     "                if gate_err:\n                    return gate_err\n",
     '                _gate_extra_flags({"policy.push_to_hub": push_to_hub}, tool_context)\n'),
    ("M3 gate unconditionally (default false too)", "lerobot_train",
     "            if push_to_hub:\n                gate_err = _gate_extra_flags(", 
     "            if True:\n                gate_err = _gate_extra_flags("),
    ("M4 drop policy.push_to_hub from the blocklist", None,
     '        "policy.push_to_hub",\n', ""),
    ("M5 gate names a flag that is not blocked", "lerobot_train",
     '_gate_extra_flags({"policy.push_to_hub": push_to_hub}, tool_context)',
     '_gate_extra_flags({"policy.optimizer_lr": push_to_hub}, tool_context)'),
]

def region(src, fn_name):
    if fn_name is None:
        return 0, len(src.splitlines())
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return fn.lineno - 1, fn.end_lineno

def run(target):
    out = subprocess.run([sys.executable, "-m", "pytest", target, "-q", "--no-cov",
                          "-p", "no:randomly", "--tb=no"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    lines = [l for l in out.splitlines() if re.match(r"^={5,}.*(passed|failed|error)", l)]
    tail = lines[-1] if lines else out.strip().splitlines()[-1:]
    m_f = re.search(r"(\d+) failed", str(tail)); m_e = re.search(r"(\d+) error", str(tail))
    m_p = re.search(r"(\d+) passed", str(tail))
    return (int(m_f.group(1)) if m_f else 0) + (int(m_e.group(1)) if m_e else 0), \
           int(m_p.group(1)) if m_p else 0

original = SRC.read_text()
print(f"{'mutation':<46} {'new file':>12}  {'pre-existing':>14}")
print("-" * 76)
fn, fp = run(NEW); on, op = run(OLD)
print(f"{'(unmutated control)':<46} {str(fn)+' failed':>12}  {str(on)+' failed':>14}")
try:
    for label, scope, old, new in MUTATIONS:
        lo, hi = region(original, scope)
        lines = original.splitlines(keepends=True)
        blob = "".join(lines[lo:hi])
        in_fn, in_file = blob.count(old), original.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn} in_file={in_file}"
        mutated = "".join(lines[:lo]) + blob.replace(old, new, 1) + "".join(lines[hi:])
        assert mutated != original
        ast.parse(mutated)
        SRC.write_text(mutated)
        nf, _ = run(NEW); of, _ = run(OLD)
        print(f"{label:<46} {str(nf)+' failed':>12}  {str(of)+' failed':>14}   (in_fn={in_fn} in_file={in_file})")
finally:
    SRC.write_text(original)
    assert SRC.read_text() == original
    print("\nrestored byte-identically")
