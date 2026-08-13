"""Mutation table for the Isaac camera-scoping parity completion (both arms)."""
import ast, pathlib, re, subprocess, sys

SRC = pathlib.Path("strands_robots/simulation/isaac/recording.py")
TEST = "tests/simulation/isaac/test_dataset_recording.py"
NEW = [
    "test_start_recording_accepts_schema_safe_camera_name",
    "test_start_recording_raw_and_schema_safe_names_are_equivalent",
    "test_start_recording_dedupes_both_spellings_of_one_camera",
]
ALIAS = "                            raw, safe = safe_to_raw[requested], requested\n"
MUTS = [
    ("M1 drop the schema-safe alias branch entirely",
     "                        elif requested in safe_to_raw:  # already schema-safe\n" + ALIAS, ""),
    ("M2 alias keeps the safe spelling as its render SOURCE",
     ALIAS, "                            raw, safe = requested, requested\n"),
    ("M3 drop the both-spellings dedup",
     "                        if safe not in selected_safe:\n",
     "                        if True:  # mutated: no dedup\n"),
    ("M4 raw request does not canonicalize to the safe key",
     "                            raw, safe = requested, raw_to_safe[requested]\n",
     "                            raw, safe = requested, requested\n"),
    ("M5 alias swaps raw/safe",
     ALIAS, "                            raw, safe = requested, safe_to_raw[requested]\n"),
]

orig = SRC.read_text()
tree = ast.parse(orig)
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "start_recording")
region = "".join(orig.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])

def run(*args):
    p = subprocess.run([sys.executable, "-m", "pytest", TEST, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no", *args],
                       capture_output=True, text=True, cwd=".")
    tail = [ln for ln in p.stdout.splitlines() if re.match(r"^={5,}.*(passed|failed|error)", ln)]
    line = tail[-1] if tail else p.stdout.strip().splitlines()[-1]
    f = int(m.group(1)) if (m := re.search(r"(\d+) failed", line)) else 0
    e = sum(int(x) for x in re.findall(r"(\d+) errors?", line))
    pa = int(m.group(1)) if (m := re.search(r"(\d+) passed", line)) else 0
    return f + e, pa

NEWK = " or ".join(NEW)
print(f"{'mutation':<52} {'new':>10} {'pre-existing':>14}")
print("-" * 80)
c_new, c_old = run("-k", NEWK), run("-k", f"not ({NEWK})")
print(f"{'(unmutated control)':<52} {c_new[0]:>3} failed {c_new[1]:>3}p {c_old[0]:>4} failed {c_old[1]:>3}p")
assert c_new[0] == 0 and c_old[0] == 0, (c_new, c_old)
try:
    for label, old, new in MUTS:
        in_fn, in_file = region.count(old), orig.count(old)
        assert in_fn == 1, (label, in_fn, in_file)
        SRC.write_text(orig.replace(old, new, 1))
        rn, ro = run("-k", NEWK), run("-k", f"not ({NEWK})")
        print(f"{label:<52} {rn[0]:>3} failed {rn[1]:>3}p {ro[0]:>4} failed {ro[1]:>3}p   [in_fn={in_fn} in_file={in_file}]")
finally:
    SRC.write_text(orig)
    assert SRC.read_text() == orig
print("\nrestored byte-identically")
