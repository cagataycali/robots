import ast, pathlib, re, shutil, subprocess, sys

SRC = pathlib.Path("strands_robots/simulation/isaac/motion_primitives.py")
NEWF = "tests/simulation/isaac/test_articulation_read_write_surfaces.py"
OLDF = "tests/simulation/isaac/test_move_to_ik.py"
NEW_K = ("TestTheBasePoseReadbackAnswersEveryUnreadableRoute or "
         "TestMoveToRefusesAnUnreadableBaseRatherThanSubstitutingTheOrigin")

original = SRC.read_text()
backup = pathlib.Path(f"/tmp/mutbak-{SRC.name}")
shutil.copy(SRC, backup)

fn = next(n for n in ast.walk(ast.parse(original))
          if isinstance(n, ast.FunctionDef) and n.name == "_articulation_base_pose")
lines = original.splitlines(keepends=True)
region = "".join(lines[fn.lineno - 1:fn.end_lineno])

RAISE_ARM = "        except (RuntimeError, ValueError, AttributeError, TypeError):\n            return None\n"
MUTATIONS = [
    ("M1 exception set narrows (a torn-down read escapes)",
     RAISE_ARM, "        except (KeyError,):\n            return None\n"),
    ("M2 the raise route substitutes an ORIGIN base",
     RAISE_ARM, "        except (RuntimeError, ValueError, AttributeError, TypeError):\n"
                "            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])\n"),
    ("M3 unpack handler narrows",
     "        except (TypeError, ValueError):\n            return None\n",
     "        except (KeyError,):\n            return None\n"),
    ("M4 drop the None-component check",
     "        if pos_raw is None or quat_raw is None:\n            return None\n", ""),
    ("M5 drop the component-count check",
     "        if pos.size != 3 or quat.size != 4:\n            return None\n", ""),
    ("M6 drop the non-finite check",
     "        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):\n            return None\n", ""),
    ("M7 drop the zero-norm check",
     "        if norm < 1e-8:\n            return None\n", ""),
    ("M8 stop normalizing the quaternion",
     "        return pos, quat / norm\n", "        return pos, quat\n"),
]

def run(args):
    p = subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "--no-cov", "-p", "no:randomly"],
                       capture_output=True, text=True)
    m = re.search(r"(\d+) failed", p.stdout)
    return int(m.group(1)) if m else 0

print(f"{'mutation':<52} {'new cases':>10} {'pre-existing':>13}")
print("-" * 78)
try:
    for label, old, new in MUTATIONS:
        in_fn, in_file = region.count(old), original.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn} in_file={in_file}"
        # Splice within the function's own line range: this anchor also
        # occurs in the sibling read surface, so a file-wide replace would
        # mutate the wrong function (in_file=2 below).
        mutated = "".join(lines[: fn.lineno - 1]) + region.replace(old, new, 1) + "".join(lines[fn.end_lineno :])
        assert mutated != original, label
        SRC.write_text(mutated)
        a = run([NEWF, "-k", NEW_K])
        b = run([NEWF, OLDF, "-k", f"not ({NEW_K})"])
        print(f"{label:<52} {a:>10} {b:>13}   (anchor in_fn={in_fn} in_file={in_file})")
        SRC.write_text(original)
    a = run([NEWF, "-k", NEW_K]); b = run([NEWF, OLDF, "-k", f"not ({NEW_K})"])
    print(f"{'(unmutated control)':<52} {a:>10} {b:>13}")
finally:
    SRC.write_text(original)
    assert SRC.read_text() == backup.read_text(), "RESTORE FAILED"
    print("\nrestored byte-identically")
