"""Mutation table: 6 plausible regressions x 2 arms (new tests vs pre-existing)."""
import ast, pathlib, subprocess, sys

SRC = pathlib.Path("strands_robots/tools/harness_memory.py")
NEW = ["tests/tools/test_harness_memory_unusable_inputs.py"]
OLD = ["tests/tools/test_harness_memory.py", "tests/tools/test_harness_memory_summary_budget.py"]

MUTATIONS = [
    ("M1 malformed-spec handler dropped", "get_valid_actions",
     '''        try:
            sim_actions = spec["properties"]["action"]["enum"]
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed simulation tool spec at {spec_path}: {e}") from e
''',
     '''        sim_actions = spec["properties"]["action"]["enum"]
'''),
    ("M2 trace serializability handler dropped", "_validate_trace",
     '''        try:
            total_bytes += len(json.dumps(entry, sort_keys=True))
        except (TypeError, ValueError) as e:
            raise ValueError(f"trace[{i}] is not JSON-serializable: {e}") from e
''',
     '''        total_bytes += len(json.dumps(entry, sort_keys=True))
'''),
    ("M3 summary serializability handler dropped", "_validate_summary",
     '''    try:
        size = len(json.dumps(summary, sort_keys=True))
    except (TypeError, ValueError) as e:
        raise ValueError(f"summary is not JSON-serializable: {e}") from e
''',
     '''    size = len(json.dumps(summary, sort_keys=True))
'''),
    ("M4 version fallback dropped", "_version_string",
     '''    try:
        return _importlib_metadata.version("strands-robots")
    except _importlib_metadata.PackageNotFoundError:
        return "unknown"
''',
     '''    return _importlib_metadata.version("strands-robots")
'''),
    ("M5 non-UTF-8 handler dropped", "_read_rules",
     '''        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"rule store at {path.name} is not valid UTF-8 ({e})") from e
''',
     '''        content = path.read_text(encoding="utf-8")
'''),
    ("M6 load_rules swallows per kind (silent partial)", "load_rules",
     '''        return {kind: self._read_rules(self.global_dir / fname) for kind, fname in _RULE_FILES.items()}
''',
     '''        out: dict[str, list[str]] = {}
        for kind, fname in _RULE_FILES.items():
            try:
                out[kind] = self._read_rules(self.global_dir / fname)
            except ValueError:
                out[kind] = []
        return out
'''),
]

original = SRC.read_text(encoding="utf-8")


def fn_range(src: str, name: str) -> tuple[int, int]:
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, (n.end_lineno or n.lineno)
    raise AssertionError(f"no function {name}")


def run(paths: list[str]) -> tuple[int, int]:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q", "-p", "no:randomly", "--no-cov", "--timeout=120"],
        capture_output=True, text=True,
    )
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    import re
    f = re.search(r"(\d+) failed", tail)
    p = re.search(r"(\d+) passed", tail)
    return (int(f.group(1)) if f else 0, int(p.group(1)) if p else 0)


print(f"{'mutation':<48} {'new tests':<16} {'pre-existing':<16}")
print("-" * 82)
rows = []
try:
    for label, fname, old, new in MUTATIONS:
        lo, hi = fn_range(original, fname)
        region = "\n".join(original.splitlines()[lo - 1:hi]) + "\n"
        in_fn, in_file = region.count(old), original.count(old)
        assert in_fn == 1, f"{label}: anchor appears {in_fn}x inside {fname} (expected 1)"
        mutated = original.replace(old, new, 1)
        assert mutated != original
        ast.parse(mutated)
        SRC.write_text(mutated, encoding="utf-8")
        nf, np_ = run(NEW)
        of, op = run(OLD)
        rows.append((label, in_fn, in_file, nf, np_, of, op))
        print(f"{label:<48} {str(nf) + ' failed':<16} {str(of) + ' failed':<16}  (anchor in_fn={in_fn} in_file={in_file})")
        SRC.write_text(original, encoding="utf-8")
finally:
    SRC.write_text(original, encoding="utf-8")

assert SRC.read_text(encoding="utf-8") == original, "restore failed"
print("-" * 82)
print(f"caught by the new tests : {sum(1 for r in rows if r[3] > 0)} of {len(rows)}")
print(f"caught by pre-existing  : {sum(1 for r in rows if r[5] > 0)} of {len(rows)}")
print("source restored byte-identical: OK")
import json as _j
pathlib.Path("/tmp/mutations.json").write_text(_j.dumps(rows, indent=2))
