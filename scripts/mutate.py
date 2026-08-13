"""Mutation table: 7 plausible regressions x 2 arms (new module vs pre-existing)."""
import os, pathlib, re, shutil, subprocess, sys

RID = os.environ["GITHUB_RUN_ID"]
ROOT = pathlib.Path(f"/tmp/robots-mine-{RID}")
SRC = ROOT / "strands_robots/tools/lerobot_teleoperate.py"
NEW = "tests/tools/test_teleop_auto_accept_reports_failure.py"
OLD = "tests/tools/test_lerobot_teleoperate.py"
SAVE = pathlib.Path(f"/tmp/mut-save-{RID}.py")
shutil.copy(SRC, SAVE)
BASE = SAVE.read_text()
lines = BASE.splitlines(keepends=True)

# Extract the real handler text from the file rather than hand-typing it.
i_exc = next(i for i, l in enumerate(lines) if l.strip() == "except Exception as exc:")
i_warn = next(i for i in range(i_exc, len(lines)) if lines[i].strip() == "logger.warning(")
i_end = next(i for i in range(i_warn, len(lines)) if lines[i].strip() == ")")
HANDLER = "".join(lines[i_exc:i_end + 1])          # except + comment + warning
WARN = "".join(lines[i_warn:i_end + 1])            # the logger.warning call only
assert BASE.count(HANDLER) == 1 and BASE.count(WARN) == 1

MUTS = [
    ("M1 revert to a bare swallow", HANDLER,
     "                        except Exception:\n                            pass\n", None),
    ("M2 keep the handler, drop only the record", WARN,
     "                            pass\n", None),
    ("M3 downgrade WARNING to DEBUG", "                            logger.warning(\n",
     "                            logger.debug(\n", None),
    ("M4 drop the reason from the record",
     "                                session_name,\n                                exc,\n",
     "                                session_name,\n", None),
    ("M5 drop the where-to-look guidance",
     '                                "the calibration prompt may be unanswered - check "\n'
     "                                \"action='status' and the session log\",\n",
     '                                "auto-accept failed",\n', None),
    ("M6 also report on success (over-reach)",
     "                            proc.stdin.close()  # Close stdin after sending responses\n",
     '                            proc.stdin.close()  # Close stdin after sending responses\n'
     '                            logger.warning("[teleop] auto-accept wrote to stdin")\n', None),
    ("M7 reword stop's refusal so it drifts from status",
     '                return {"status": "error", "content": [{"text": f"Session \'{session_name}\' not found"}]}\n',
     '                return {"status": "error", "content": [{"text": f"no such session {session_name}"}]}\n',
     (1115, 1162)),
]

def run(paths, extra):
    r = subprocess.run([sys.executable, "-m", "pytest", *paths, *extra, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "MUJOCO_GL": "egl"})
    tail = [l for l in r.stdout.splitlines() if re.search(r"^=+.*(passed|failed|error)", l)]
    line = tail[-1] if tail else "".join(r.stdout.strip().splitlines()[-1:])
    f = re.search(r"(\d+) failed", line); e = re.search(r"(\d+) errors?", line)
    return (int(f.group(1)) if f else 0) + (int(e.group(1)) if e else 0)

NEW_NAMES = re.findall(r"^def (test_\w+)", (ROOT / NEW).read_text(), re.M)
DESELECT = ["-k", "not (" + " or ".join(NEW_NAMES) + ")"]
print(f"pre-existing arm = {OLD} with the {len(NEW_NAMES)} new tests deselected by name\n")
print(f"{'mutation':<46} {'new':>5} {'pre-existing':>13}")
print("-" * 68)
print(f"{'(unmutated control)':<46} {run([NEW], []):>5} {run([NEW, OLD], DESELECT):>13}")
try:
    for label, old, new, rng in MUTS:
        if rng:
            ls = BASE.splitlines(keepends=True); lo, hi = rng
            region = "".join(ls[lo - 1:hi])
            print(f"    [{label[:2]}] anchor in_range={region.count(old)} in_file={BASE.count(old)}"
                  f"  <- line-range scoping load-bearing")
            assert region.count(old) == 1
            ls[lo - 1:hi] = [region.replace(old, new, 1)]
            out = "".join(ls)
        else:
            assert BASE.count(old) == 1, f"{label}: in_file={BASE.count(old)}"
            out = BASE.replace(old, new, 1)
        assert out != BASE
        SRC.write_text(out)
        print(f"{label:<46} {run([NEW], []):>5} {run([NEW, OLD], DESELECT):>13}")
finally:
    shutil.copy(SAVE, SRC)
    assert SRC.read_text() == BASE
    print("\nsource restored byte-identically")
