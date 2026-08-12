"""Mutation table: 8 plausible regressions x 2 arms (new module / pre-existing suite)."""
import ast, json, pathlib, shutil, subprocess, sys, tempfile

SRC = pathlib.Path("strands_robots/simulation/isaac/motion_primitives.py")
NEW = "tests/simulation/isaac/test_articulation_read_write_surfaces.py"
OLD = "tests/simulation/isaac/test_motion_primitives.py"

# (label, enclosing function, old, new)
MUTATIONS = [
    ("M1 the get_dof_limits fallback source is dropped", "_articulation_dof_limits",
     '            get_limits = getattr(articulation, "get_dof_limits", None)\n',
     '            get_limits = None\n'),
    ("M2 an unreadable hasLimits field is not tolerated", "_articulation_dof_limits",
     '                except (KeyError, ValueError, IndexError, TypeError):\n                    has_limits = None\n',
     '                except (KeyError, ValueError, IndexError, TypeError):\n                    raise\n'),
    ("M3 an unreadable position read substitutes zeros", "_read_joint_positions",
     '        except (RuntimeError, ValueError, AttributeError, TypeError):\n            return None\n',
     '        except (RuntimeError, ValueError, AttributeError, TypeError):\n            return np.zeros(1, dtype=np.float64)\n'),
    ("M4 a failed target write is swallowed as success", "_apply_position_targets",
     '            return _err(f"{action}: failed to set joint position targets on \'{robot_name}\': {e}")\n',
     '            return None\n'),
    ("M5 set_gripper fabricates the readback it could not do", "set_gripper",
     '                q = self._read_joint_positions(articulation)\n',
     '                q = self._read_joint_positions(articulation)\n'
     '                q = np.zeros(len(short_names)) if q is None else q\n'),
    ("M6 rotate_wrist ignores a mid-run read failure", "rotate_wrist",
     '                    return _err(f"rotate_wrist: could not read joint positions from \'{name}\' mid-run; aborting.")\n',
     '                    break\n'),
    ("M7 a degenerate span is accepted", "_articulation_dof_limits",
     '            if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:\n',
     '            if not (math.isfinite(lo) and math.isfinite(hi)):\n'),
    ("M8 a non-finite bound is accepted", "_articulation_dof_limits",
     '            if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:\n',
     '            if hi <= lo:\n'),
]


def fn_range(src, name):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(f"no function {name}")


def run(target, label):
    out = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--no-header", "-p", "no:randomly", "--no-cov", "--tb=no"],
        capture_output=True, text=True, env={**__import__("os").environ, "MUJOCO_GL": "egl"},
    ).stdout
    import re
    f = re.search(r"(\d+) failed", out)
    p = re.search(r"(\d+) passed", out)
    return (int(f.group(1)) if f else 0), (int(p.group(1)) if p else 0)

original = SRC.read_text()
rows = []
try:
    for label, fname, old, new in MUTATIONS:
        lo, hi = fn_range(original, fname)
        region = "\n".join(original.splitlines()[lo - 1:hi]) + "\n"
        in_fn, in_file = region.count(old), original.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn} (in_file={in_file})"
        mutated = region.replace(old, new)
        full = original.replace(region, mutated)
        assert full != original, label
        SRC.write_text(full)
        nf, np_ = run(NEW, label)
        of, op = run(OLD, label)
        rows.append({"label": label, "in_fn": in_fn, "in_file": in_file,
                     "new_failed": nf, "new_passed": np_, "old_failed": of, "old_passed": op})
        print(f"{label:52s} in_fn={in_fn}/in_file={in_file}  new: {nf} failed  |  pre-existing: {of} failed / {op} passed")
        SRC.write_text(original)
finally:
    SRC.write_text(original)
    assert SRC.read_text() == original, "restore FAILED"

# unmutated control
nf, np_ = run(NEW, "control"); of, op = run(OLD, "control")
print(f"\n{'UNMUTATED control':52s} new: {nf} failed / {np_} passed  |  pre-existing: {of} failed / {op} passed")
caught_new = sum(1 for r in rows if r["new_failed"] > 0)
caught_old = sum(1 for r in rows if r["old_failed"] > 0)
print(f"\ncaught by the new module: {caught_new}/{len(rows)}   caught by the pre-existing suite: {caught_old}/{len(rows)}")
json.dump({"rows": rows, "control": {"new_passed": np_, "old_passed": op},
           "caught_new": caught_new, "caught_old": caught_old},
          open(f"/tmp/mut-{__import__('os').environ['GITHUB_RUN_ID']}.json", "w"), indent=2)
print("source restored byte-identically:", SRC.read_text() == original)
