"""Mutation table: 6 regressions x 2 arms. Anchors are AST-scoped per function."""
import ast, pathlib, re, shutil, subprocess, sys, tempfile

SRC = pathlib.Path("strands_robots/tools/lerobot_train.py")
NEW_CLASS = "TestWhichAllowlistEntryClearsWhichSpelling"
FILE = "tests/tools/test_train_publication_requires_approval.py"

MAIN_ENTRY = """        push_to_hub: Push the trained checkpoint to the HF Hub at the end.
            Publishing is an outward-facing action, so a true value requires
            operator approval through ``tool_context`` (or an explicit
            STRANDS_TRAIN_EXTRA_FLAGS_ALLOW entry) exactly as the
            ``extra_flags={'push_to_hub': True}`` spelling already does. The
            default false value emits the flag unchanged and is not gated.
"""

MUTATIONS = [
    ("M1 revert the description to main's wording", None, "__ENTRY__", MAIN_ENTRY),
    ("M2 gate the named parameter under the bare key",
     "lerobot_train", '{"policy.push_to_hub": push_to_hub}', '{"push_to_hub": push_to_hub}'),
    ("M3 drop policy.push_to_hub from the blocklist", None, '        "policy.push_to_hub",\n', ""),
    ("M4 drop push_to_hub from the blocklist", None, '        "push_to_hub",\n', ""),
    ("M5 description names the bare key instead", None,
     "STRANDS_TRAIN_EXTRA_FLAGS_ALLOW=policy.push_to_hub", "STRANDS_TRAIN_EXTRA_FLAGS_ALLOW=push_to_hub"),
    ("M6 remove the explanatory comment on the blocklist pair", None,
     "        # Blocked in both spellings, and the pair is not a duplicate: the named\n", ""),
]


def fn_range(src: str, name: str) -> tuple[int, int]:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node.lineno, node.end_lineno
    raise AssertionError(name)


def run(*sel: str) -> str:
    p = subprocess.run([sys.executable, "-m", "pytest", FILE, *sel, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no"], capture_output=True, text=True)
    tail = [ln for ln in p.stdout.splitlines() if re.match(r"^={5,}.*(passed|failed|error)", ln)]
    out = tail[-1] if tail else p.stdout.strip().splitlines()[-1]
    f = re.search(r"(\d+) (?:failed|error)", out)
    return f"{f.group(1) if f else 0} failed  |  {out.strip('= ')}"


def entry_block(src: str) -> str:
    lo, hi = fn_range(src, "lerobot_train")
    lines = src.splitlines(keepends=True)
    i = next(k for k in range(lo - 1, hi) if lines[k].lstrip().startswith("push_to_hub: Push"))
    j = next(k for k in range(i + 1, hi) if lines[k].lstrip().startswith("resume:"))
    return "".join(lines[i:j])


original = SRC.read_text()
backup = pathlib.Path(tempfile.mkdtemp()) / "orig.py"
shutil.copy(SRC, backup)
try:
    print(f"{'mutation':<52} {'new class':<34} pre-existing")
    print(f"{'(unmutated control)':<52} {run('-k', NEW_CLASS):<34} {run('-k', f'not {NEW_CLASS}')}")
    for label, scope, old, new in MUTATIONS:
        src = original
        if old == "__ENTRY__":
            old = entry_block(src)
        if scope:
            lo, hi = fn_range(src, scope)
            region = "".join(src.splitlines(keepends=True)[lo - 1 : hi])
            in_fn, in_file = region.count(old), src.count(old)
            assert in_fn == 1, f"{label}: in_fn={in_fn}"
            print(f"    [{label[:2]} anchor in_fn={in_fn} in_file={in_file}]")
        else:
            assert src.count(old) == 1, f"{label}: {src.count(old)} occurrences"
        SRC.write_text(src.replace(old, new, 1))
        a, b = run("-k", NEW_CLASS), run("-k", f"not {NEW_CLASS}")
        print(f"{label:<52} {a:<34} {b}")
        SRC.write_text(original)
finally:
    shutil.copy(backup, SRC)
    assert SRC.read_text() == original, "restore failed"
    print("\nsource restored byte-identically")
