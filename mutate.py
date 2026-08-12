"""Mutation table: 6 plausible regressions x 2 arms (new file vs pre-existing)."""
import ast, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "strands_robots/policies/cosmos3/policy_diffusers.py"
NEW = "tests/policies/cosmos3/test_native_stack_absent.py"
OLD = "tests/policies/cosmos3/test_policy_diffusers.py"
ORIG = SRC.read_text()

def fn_range(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(f"function {name!r} not found")

def apply(name, old, new):
    """Replace `old`->`new` inside function `name` only. Prints in_fn/in_file."""
    src = ORIG
    lo, hi = fn_range(src, name)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo-1:hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"{name}: anchor appears {in_fn}x in fn (need 1); {in_file}x in file"
    print(f"    anchor in_fn={in_fn} in_file={in_file}")
    lines[lo-1:hi] = [region.replace(old, new, 1)]
    out = "".join(lines)
    ast.parse(out)
    SRC.write_text(out)

def append(text):
    out = ORIG.rstrip("\n") + "\n\n\n" + text
    ast.parse(out)
    SRC.write_text(out)

def run(paths):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q", "--no-header", "-p", "no:randomly", "--no-cov", "--tb=no"],
        cwd=ROOT, capture_output=True, text=True,
    )
    import re
    f = re.search(r"(\d+) failed", r.stdout)
    p = re.search(r"(\d+) passed", r.stdout)
    return (int(f.group(1)) if f else 0), (int(p.group(1)) if p else 0)

MUTS = [
    ("M1 _as_action_tensor degrades instead of refusing", lambda: apply(
        "_as_action_tensor",
        "        try:\n            import torch\n        except ImportError as e:\n            raise ImportError(_install_hint()) from e\n",
        "        import torch\n")),
    ("M2 _as_action_tensor rewords the remedy locally", lambda: apply(
        "_as_action_tensor",
        "raise ImportError(_install_hint()) from e",
        'raise ImportError("torch is required") from e')),
    ("M3 _as_action_tensor drops the cause chaining", lambda: apply(
        "_as_action_tensor",
        "raise ImportError(_install_hint()) from e",
        "raise ImportError(_install_hint()) from None")),
    ("M4 _to_numpy refuses instead of degrading", lambda: apply(
        "_to_numpy",
        "        except ImportError:\n            pass\n",
        "        except ImportError as e:\n            raise ImportError(_install_hint()) from e\n")),
    ("M5 _to_numpy drops the half-precision up-cast", lambda: apply(
        "_to_numpy",
        "            if isinstance(value, torch.Tensor) and value.dtype in (torch.bfloat16, torch.float16):\n                value = value.to(torch.float32)\n",
        "            pass\n")),
    ("M6 a fourth install-hint site ships undriven", lambda: append(
        "def _load_tokenizer() -> object:\n"
        '    """A new lazy native import reporting the shared remedy."""\n'
        "    try:\n        import transformers\n    except ImportError as e:\n"
        "        raise ImportError(_install_hint()) from e\n    return transformers\n")),
]

print("=" * 92)
print(f"{'mutation':<48} {'new file':>16} {'pre-existing':>18}")
print("=" * 92)
rows = []
try:
    nf, np_ = run([NEW]); of, op = run([OLD])
    print(f"{'(unmutated control)':<48} {f'{nf} failed/{np_} pass':>16} {f'{of} failed/{op} pass':>18}")
    for label, fn in MUTS:
        print(f"  applying: {label}")
        try:
            fn()
        except AssertionError as e:
            print(f"    SKIP: {e}"); SRC.write_text(ORIG); continue
        nf, np_ = run([NEW]); of, op = run([OLD])
        SRC.write_text(ORIG)
        rows.append({"label": label, "new_failed": nf, "old_failed": of})
        print(f"{label:<48} {f'{nf} failed/{np_} pass':>16} {f'{of} failed/{op} pass':>18}")
finally:
    SRC.write_text(ORIG)
    assert SRC.read_text() == ORIG, "RESTORE FAILED"
    print("=" * 92)
    print("source restored byte-identically:", SRC.read_text() == ORIG)
    caught_new = sum(1 for r in rows if r["new_failed"] > 0)
    caught_old = sum(1 for r in rows if r["old_failed"] > 0)
    print(f"caught by the new module: {caught_new} of {len(rows)}")
    print(f"caught by the pre-existing suite: {caught_old} of {len(rows)}")
    pathlib.Path(f"/tmp/mut-{sys.argv[1] if len(sys.argv)>1 else 'x'}.json").write_text(json.dumps(rows, indent=2))
