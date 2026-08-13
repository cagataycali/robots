import os, pathlib, re, subprocess, sys

SRC = pathlib.Path("strands_robots/simulation/safe_output.py")
TESTS = ["tests/simulation/test_output_path_sandbox.py",
         "tests/simulation/mujoco/test_render_output_path_roundtrip.py"]
orig = SRC.read_text()
L = orig.splitlines(keepends=True)

def idx(pred):
    return next(i for i, l in enumerate(L) if pred(l))

i_trav  = idx(lambda l: '".." in raw.parts' in l)          # 2-line block
i_anch  = idx(lambda l: l.startswith("    # A bare filename"))
i_anch_end = idx(lambda l: "raw = sandbox_root / raw" in l)
i_symc  = idx(lambda l: "Refuse to follow a symlink planted" in l)
i_sym_end = idx(lambda l: "refusing to follow" in l)

TRAV = L[i_trav:i_trav+2]                 # if .. / raise
ANCH = L[i_anch:i_anch_end+1]             # comment block + if + assign
SYMB = L[i_symc:i_sym_end+1]              # comment + if + raise
assert i_trav + 2 == i_anch and i_anch_end + 1 == i_symc, "layout moved"
print(f"  blocks: TRAV={len(TRAV)}L ANCH={len(ANCH)}L SYM={len(SYMB)}L  (lines {i_trav+1}..{i_sym_end+1})")

def rebuild(order):
    out = L[:i_trav] + [x for blk in order for x in blk] + L[i_sym_end+1:]
    return "".join(out)

NEW_NAMES = [
    "test_sandbox_anchors_bare_filename_to_root",
    "test_bare_filename_anchoring_ignores_cwd",
    "test_sandbox_still_rejects_relative_path_with_separator",
    "test_sandbox_rejects_bare_dotdot_before_anchoring",
    "test_sandbox_refuses_symlink_planted_at_anchored_destination",
    "test_anchored_bare_filename_still_rejects_metacharacters",
    "test_guards_only_leaves_bare_filename_cwd_relative",
    "test_allow_abs_leaves_bare_filename_cwd_relative",
    "test_render_bare_filename_writes_into_the_sandbox",
]
K_NEW = " or ".join(NEW_NAMES)
K_OLD = "not (" + K_NEW + ")"

MUTS = [
    ("M1 delete the anchoring (revert the fix)", rebuild([TRAV, SYMB])),
    ("M2 anchor BEFORE the traversal guard",     rebuild([ANCH, TRAV, SYMB])),
    ("M3 anchor AFTER the symlink probe",        rebuild([TRAV, SYMB, ANCH])),
    ("M4 widen: anchor ANY relative path",
     orig.replace("and not raw.is_absolute() and len(raw.parts) == 1:", "and not raw.is_absolute():", 1)),
    ("M5 anchor even when allow_abs opted out",
     orig.replace("if sandbox_root is not None and not allow_abs and not raw.is_absolute()",
                  "if sandbox_root is not None and not raw.is_absolute()", 1)),
]

def run(k):
    r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-cov", "-p", "no:randomly", "-k", k],
                       capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    f = re.search(r"(\d+) failed", r.stdout)
    p = re.search(r"(\d+) passed", r.stdout)
    return (int(f.group(1)) if f else 0), (int(p.group(1)) if p else 0)

print()
print(f"{'mutation':46} {'new (9)':>12} {'pre-existing':>14}")
print("-" * 74)
try:
    for name, mutated in MUTS:
        assert mutated != orig, f"{name}: no-op"
        SRC.write_text(mutated)
        fn, _ = run(K_NEW)
        fo, po = run(K_OLD)
        print(f"{name:46} {str(fn)+' failed':>12} {str(fo)+' failed':>14}")
        SRC.write_text(orig)
    fn, pn = run(K_NEW); fo, po = run(K_OLD)
    print(f"{'(control: unmutated)':46} {str(fn)+' failed':>12} {str(fo)+' failed':>14}   [{pn} + {po} pass]")
finally:
    SRC.write_text(orig)
    assert SRC.read_text() == orig
    print("\nrestored byte-identical: OK")
