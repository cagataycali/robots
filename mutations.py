import os, pathlib, re, subprocess, sys

ROOT = pathlib.Path.cwd()
NEW = ROOT / "tests/test_mujoco_render_assertions_are_gl_gated.py"
OLD = ROOT / "tests/test_zz_old_guard_probe.py"
BASE = "upstream/main:tests/test_mujoco_render_assertions_are_gl_gated.py"

def run(path):
    p = subprocess.run(
        [sys.executable, "-m", "pytest", str(path.relative_to(ROOT)), "-q", "--no-cov", "-p", "no:randomly"],
        capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
    line = [l for l in p.stdout.splitlines() if ("passed" in l or "failed" in l) and "=" in l]
    names = sorted({m.split("::")[-1] for m in re.findall(r"FAILED \S+", p.stdout)})
    names = sorted({l.split("::")[-1].split(" ")[0] for l in p.stdout.splitlines() if l.startswith("FAILED")})
    return (line[-1].strip("= ") if line else "?"), names

old_src = subprocess.run(["git","show",BASE],capture_output=True,text=True,check=True).stdout
new_src = NEW.read_text(encoding="utf-8")
HELPER_OLD = "    return set(expected) - set(gated), set(gated) - set(expected)"

VACUITY = {
  "V1 survey finds nothing": ("    return ungated, gated, other_backend", "    return {}, {}, other_backend"),
  "V2 nothing classified gated": ("            bucket = gated if is_gated(tree, lineno, names) else ungated",
                                  "            bucket = ungated"),
  "V3 scan rooted at a subdirectory": ('    return pathlib.Path(__file__).parent',
                                       '    return pathlib.Path(__file__).parent / "policies"'),
}
# M1 now models the OLD rule faithfully: the verdict IS the count comparison.
HELPER = {
  "M1 the pin compares counts again":
    "    counted = sum(len(v) for v in gated.values())\n"
    '    return (set() if counted == len(expected) else {f"<count {counted} != {len(expected)}>"}), set()',
  "M2 the missing half is dropped": "    return set(), set(gated) - set(expected)",
  "M3 the unexpected half is dropped": "    return set(expected) - set(gated), set()",
}

NEWPIN = "test_every_module_the_survey_covers_contributes_a_gated_assertion"
OLDPINS = {"test_the_survey_covers_the_modules_it_is_meant_to", "test_every_in_scope_assertion_is_accounted_for"}

OLD.write_text(old_src, encoding="utf-8")
try:
    for label, (a, r) in [("(unmutated)", (None, None))] + list(VACUITY.items()):
        if a is None:
            OLD.write_text(old_src, encoding="utf-8"); NEW.write_text(new_src, encoding="utf-8")
        else:
            assert old_src.count(a) == 1 and new_src.count(a) == 1, label
            OLD.write_text(old_src.replace(a, r, 1), encoding="utf-8")
            NEW.write_text(new_src.replace(a, r, 1), encoding="utf-8")
        try:
            os_, on = run(OLD); ns, nn = run(NEW)
            print(f"{label}\n    old: {os_:28} caught by old pins: {sorted(set(on) & OLDPINS)}")
            print(f"    new: {ns:28} caught by the new pin: {NEWPIN in nn}")
        finally:
            OLD.write_text(old_src, encoding="utf-8"); NEW.write_text(new_src, encoding="utf-8")
    assert new_src.count(HELPER_OLD) == 1
    for label, r in HELPER.items():
        NEW.write_text(new_src.replace(HELPER_OLD, r, 1), encoding="utf-8")
        try:
            ns, nn = run(NEW)
            print(f"{label}\n    new: {ns:28} failing: {nn}")
        finally:
            NEW.write_text(new_src, encoding="utf-8")
finally:
    OLD.unlink(missing_ok=True)
assert NEW.read_text(encoding="utf-8") == new_src and not OLD.exists()
print("\nrestored:", subprocess.run(["git","status","--porcelain"],capture_output=True,text=True).stdout.strip())
