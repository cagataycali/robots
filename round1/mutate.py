import ast, pathlib, re, subprocess, sys
sys.path.insert(0, "_probe")
from mirror import unused_locals

SRC  = pathlib.Path("strands_robots/tools/lerobot_teleoperate.py")
TEST = pathlib.Path("tests/tools/test_teleop_auto_accept_reports_failure.py")
ARM_A = [str(TEST)]
ARM_B = ["tests/tools/test_lerobot_teleoperate.py",
         "tests/tools/test_lerobot_teleoperate_flag_domain.py"]

def fn_range(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(f"no function {name}")

def scoped(src, name, old, new):
    """Replace `old` with `new`, asserting it is unique inside function `name`."""
    lo, hi = fn_range(src, name)
    lines = src.splitlines(keepends=True)
    region = "".join(lines[lo - 1:hi])
    in_fn, in_file = region.count(old), src.count(old)
    assert in_fn == 1, f"anchor in_fn={in_fn} in_file={in_file} for {name}"
    print(f"    anchor in_fn={in_fn} in_file={in_file}")
    return "".join(lines[:lo - 1]) + region.replace(old, new, 1) + "".join(lines[hi:])

REPORT = '''                        except Exception as exc:
                            # Report it. The start result above has already told the
                            # caller the session started, so this record is the only
                            # signal that the prompt went unanswered. Every other
                            # handler in this tool reports its failure - one even
                            # surfaces a log-read failure into the caller's content -
                            # and the write this guards is the whole job of
                            # ``auto_accept_calibration``. WARNING rather than DEBUG
                            # because the visible report is a success.
                            logger.warning(
                                "[teleop] session %r: auto-accept did not complete (%s); "
                                "the calibration prompt may be unanswered - check "
                                "action='status' and the session log",
                                session_name,
                                exc,
                            )
'''
SWALLOW = '''                        except Exception:
                            pass  # Ignore errors if process has already finished
'''
WRITES = '''                            proc.stdin.write("\\n")  # Send ENTER
                            proc.stdin.flush()
                            time.sleep(1)
                            proc.stdin.write("\\n")  # Send another ENTER (for robot calibration)
                            proc.stdin.flush()
'''

MUTATIONS = {
    "M1 revert the report to a bare swallow":
        ("src", lambda s: scoped(s, "auto_respond", REPORT, SWALLOW)),
    "M2 downgrade WARNING -> DEBUG":
        ("src", lambda s: scoped(s, "auto_respond", "logger.warning(", "logger.debug(")),
    "M3 drop the session name from the record":
        ("src", lambda s: scoped(s, "auto_respond",
            '"[teleop] session %r: auto-accept did not complete (%s); "\n', '"[teleop] auto-accept did not complete (%s); "\n')
            .replace("                                session_name,\n", "", 1)),
    "M4 drop the next-step guidance":
        ("src", lambda s: scoped(s, "auto_respond",
            '"the calibration prompt may be unanswered - check "\n                                "action=\'status\' and the session log",\n',
            '"it did not complete",\n')),
    "M5 healthy auto-accept stops reporting success":
        ("src", lambda s: s.replace(
            "                    threading.Thread(target=auto_respond, daemon=True).start()\n",
            "                    threading.Thread(target=auto_respond, daemon=True).start()\n"
            '                    return {"status": "error", "content": [{"text": "mutated"}]}\n', 1)),
    "M6 happy path writes nothing":
        ("src", lambda s: scoped(s, "auto_respond", WRITES, "")),
    "M7 revert this round (restore the discarded tuple)":
        ("test", lambda s: s.replace(
            "        _start_with_auto_accept(monkeypatch, fail=False)\n",
            "        _result, _proc = _start_with_auto_accept(monkeypatch, fail=False)\n", 1)),
}

def run(paths):
    r = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no"], capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if re.match(r"^={5,}.*(passed|failed|error)", l)]
    txt = tail[-1] if tail else r.stdout.strip().splitlines()[-1]
    def g(w):
        m = re.search(rf"(\d+) {w}", txt)
        return int(m.group(1)) if m else 0
    return g("failed") + g("error") + g("errors"), g("passed")

orig_src, orig_test = SRC.read_text(), TEST.read_text()
print(f"TREE: {pathlib.Path.cwd()}")
print(f"{'mutation':46s} {'new file':>18s} {'pre-existing':>18s} {'mirror':>8s}")
fa, pa = run(ARM_A); fb, pb = run(ARM_B)
print(f"{'(unmutated control)':46s} {f'{fa}f/{pa}p':>18s} {f'{fb}f/{pb}p':>18s} {len(unused_locals(orig_test)):>8d}")
try:
    for label, (which, fn) in MUTATIONS.items():
        print(f"  {label}")
        tgt = SRC if which == "src" else TEST
        base = orig_src if which == "src" else orig_test
        mutated = fn(base)
        assert mutated != base, "mutation was a no-op"
        ast.parse(mutated)
        tgt.write_text(mutated)
        try:
            fa, pa = run(ARM_A); fb, pb = run(ARM_B)
            mir = len(unused_locals(TEST.read_text()))
            print(f"{label:46s} {f'{fa}f/{pa}p':>18s} {f'{fb}f/{pb}p':>18s} {mir:>8d}")
        finally:
            tgt.write_text(base)
finally:
    SRC.write_text(orig_src); TEST.write_text(orig_test)
    assert SRC.read_text() == orig_src and TEST.read_text() == orig_test, "RESTORE FAILED"
    print("restored byte-identically")
