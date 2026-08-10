import ast, pathlib, subprocess, sys

SRC = pathlib.Path("strands_robots/simulation/base.py")
NEW = ["tests/simulation/test_eval_facade_preflight_refusals.py"]
EXISTING = [
    "tests/simulation/test_video_config_validation.py",
    "tests/simulation/test_policy_kwargs_forwarding.py",
    "tests/simulation/test_recording_rate_matches_control_frequency.py",
    "tests/simulation/test_benchmark_horizon_domain.py",
    "tests/simulation/test_evaluate_benchmark_video.py",
    "tests/simulation/test_eval_policy_video.py",
]
ORIG = SRC.read_text()

# (label, enclosing function, the guard line to delete -- its `return err` follows)
MUTATIONS = [
    ("evaluate_benchmark drops the video guard", "evaluate_benchmark",
     'if err := self._validate_video_config(video, "evaluate_benchmark"):'),
    ("evaluate_benchmark drops the policy_config guard", "evaluate_benchmark",
     'if err := self._validate_policy_mapping(policy_config, "policy_config", "evaluate_benchmark"):'),
    ("evaluate_benchmark drops the policy_kwargs guard", "evaluate_benchmark",
     'if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "evaluate_benchmark"):'),
    ("evaluate_benchmark drops the dataset-rate guard", "evaluate_benchmark",
     'if err := self._validate_recording_rate(control_frequency, "evaluate_benchmark"):'),
    ("eval_policy drops the policy_kwargs guard", "eval_policy",
     'if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "eval_policy"):'),
    ("eval_policy drops the dataset-rate guard", "eval_policy",
     'if err := self._validate_recording_rate(control_frequency, "eval_policy"):'),
]


def fn_range(src: str, name: str) -> tuple[int, int]:
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(f"{name} not found")


def run(files: list[str]) -> tuple[int, int]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q", "--no-cov", "-p", "no:randomly", "--timeout=180"],
        capture_output=True, text=True, cwd=".",
    )
    out = p.stdout
    failed = sum(1 for line in out.splitlines() if line.startswith("FAILED"))
    return failed, p.returncode


rows = []
try:
    for label, fname, anchor in MUTATIONS:
        lo, hi = fn_range(ORIG, fname)
        lines = ORIG.splitlines(keepends=True)
        hits_fn = [i for i in range(lo - 1, hi) if lines[i].strip() == anchor]
        hits_file = [i for i, ln in enumerate(lines) if ln.strip() == anchor]
        assert len(hits_fn) == 1, f"{label}: {len(hits_fn)} hits inside {fname}"
        i = hits_fn[0]
        assert lines[i + 1].strip() == "return err", f"{label}: next line is {lines[i + 1]!r}"
        print(f"  anchor in_fn={len(hits_fn)} in_file={len(hits_file)}  ({label})")
        SRC.write_text("".join(lines[:i] + lines[i + 2:]))
        ast.parse(SRC.read_text())
        f_new, _ = run(NEW)
        f_old, _ = run(EXISTING)
        rows.append((label, f_new, f_old))
        SRC.write_text(ORIG)
finally:
    SRC.write_text(ORIG)

assert SRC.read_text() == ORIG, "source not restored"
print("\n| mutation | new tests failing | existing suite failing |")
print("|---|---|---|")
for label, f_new, f_old in rows:
    print(f"| {label} | {f_new} | {f_old} |")
assert all(r[1] > 0 for r in rows), "a mutation slipped past the new tests"
print(f"\nall {len(rows)} mutations caught by the new tests; "
      f"{sum(1 for r in rows if r[2] == 0)} of {len(rows)} invisible to the existing suite")
print("source restored byte-identical:", SRC.read_text() == ORIG)
