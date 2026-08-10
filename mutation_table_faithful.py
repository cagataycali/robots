import ast, os, pathlib, subprocess, sys
run = os.environ["GITHUB_RUN_ID"]
MINE = pathlib.Path(f"/tmp/robots-mine-{run}"); os.chdir(MINE)
SIM = pathlib.Path("strands_robots/simulation/mujoco/simulation.py")
NEW = "tests/simulation/mujoco/test_add_camera_pose_rule_has_one_owner.py"
saved = SIM.read_text()

def run_pytest(paths, extra=()):
    r = subprocess.run([sys.executable,"-m","pytest",*paths,"-q","--no-cov","-p","no:randomly",*extra],
                       capture_output=True,text=True,env={**os.environ,"MUJOCO_GL":"egl"})
    return r.stdout, r.returncode

try:
    # --- who ELSE pins the degenerate-orientation refusal? -----------------
    s = saved.replace("        if all(abs(pos[i] - tgt[i]) < 1e-9 for i in range(3)):\n",
                      "        if False:\n", 1)
    assert s != saved
    SIM.write_text(s)
    out, _ = run_pytest(["tests/simulation/mujoco","--tb=no","-q"], extra=("-x","--co","-q"))
    out, _ = run_pytest(["tests/simulation/mujoco","--tb=no"])
    fails = [l for l in out.splitlines() if l.startswith("FAILED")]
    print("### M3: which tests catch the degenerate-orientation deletion?")
    for l in fails: print("   ", l[:150])
    SIM.write_text(saved)

    # --- M1 faithful: re-add the ORIGINAL loop, verbatim -------------------
    print("\n### M1 (faithful): the original loop, re-inserted verbatim")
    anchor = "        tgt = [0.0, 0.0, 0.0] if target is None else target\n"
    assert saved.count(anchor) == 1
    loop = (
        '        for _lbl, _vec in (("position", pos), ("target", tgt)):\n'
        '            if (e := pose_vector_error("add_camera", _lbl, _vec, 3)) is not None:\n'
        '                return {"status": "error", "content": [{"text": e}]}\n'
    )
    imp = "    positive_finite_number_error,\n"
    assert saved.count(imp) == 1
    s = saved.replace(anchor, anchor + loop, 1).replace(imp, "    pose_vector_error,\n" + imp, 1)
    ast.parse(s); SIM.write_text(s)
    a, rca = run_pytest([NEW, "--tb=no"])
    b, rcb = run_pytest(["tests/simulation/mujoco","--tb=no"])
    la = [l for l in a.splitlines() if " passed" in l or " failed" in l][-1]
    lb = [l for l in b.splitlines() if " passed" in l or " failed" in l][-1]
    print(f"    new module              : {la.strip()}  -> {'CAUGHT' if rca else 'MISSED'}")
    print(f"    tests/simulation/mujoco : {lb.strip()}  -> {'CAUGHT' if rcb else 'MISSED'}")
    for l in b.splitlines():
        if l.startswith("FAILED"): print("      ", l[:140])
finally:
    SIM.write_text(saved)
    print(f"\nrestore byte-identical: {SIM.read_text() == saved}")
