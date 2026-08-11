"""Capture every measured fact this PR's figure renders. Dumps JSON.

Run from the branch checkout:  PYTHONPATH=. GITHUB_RUN_ID=<id> python3 capture.py
Reads the coverage JSONs produced by the before/after subset arms and by the
pristine-base and branch full-suite runs.
"""
import json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(".").resolve()
print("TREE:", ROOT)
RUN = os.environ["GITHUB_RUN_ID"]

NEW = ["tests/simulation/test_recording_preflight_refusals_across_backends.py"]
OLD = [
    "tests/simulation/test_dataset_recording_fps_contract.py",
    "tests/simulation/test_recording_posture_flag_domain.py",
    "tests/simulation/test_camera_name_list_contract.py",
    "tests/simulation/test_recording_rate_matches_control_frequency.py",
    "tests/simulation/isaac/test_dataset_recording.py",
    "tests/simulation/newton/test_dataset_recording.py",
]

FPS = ('        if error := dataset_recording_option_error("start_recording", fps):\n'
       '            return error\n')
POST = ('        for _flag, _value in (("push_to_hub", push_to_hub), ("overwrite", overwrite)):\n'
        '            if error := dataset_recording_posture_error("start_recording", _flag, _value):\n'
        '                return error\n')
CAMS = ('        if cameras and (text := name_list_error(cameras, "cameras", "start_recording")):\n'
        '            return {"status": "error", "content": [{"text": text}]}\n')

MUT = []
for backend in ("isaac", "newton"):
    f = f"strands_robots/simulation/{backend}/recording.py"
    MUT += [
        (backend, "DELETE the fps guard", f, FPS, ""),
        (backend, "DISCARD the fps refusal", f, FPS, FPS.replace("            return error", "            pass")),
        (backend, "DISCARD the posture refusal", f, POST,
         POST.replace("                return error", "                pass")),
        (backend, "DISCARD the cameras refusal", f, CAMS,
         CAMS.replace('            return {"status": "error", "content": [{"text": text}]}', "            pass")),
    ]


def run(files):
    out = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q", "--no-cov", "-p", "no:randomly", "--no-header"],
        capture_output=True, text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "MUJOCO_GL": "egl"},
    ).stdout
    f = re.search(r"(\d+) failed", out)
    p = re.search(r"(\d+) passed", out)
    return {"failed": int(f.group(1)) if f else 0, "passed": int(p.group(1)) if p else 0}


facts = {"tree": str(ROOT)}
facts["baseline"] = {"new": run(NEW), "pre_existing": run(OLD)}
print("baseline:", facts["baseline"])

rows = []
for backend, label, path, old, new in MUT:
    p = ROOT / path
    src = p.read_text()
    n = src.count(old)
    assert n == 1, f"{backend}/{label}: anchor appears {n}x"
    try:
        p.write_text(src.replace(old, new, 1))
        r = {"backend": backend, "label": label, "new": run(NEW), "pre_existing": run(OLD)}
    finally:
        p.write_text(src)
        assert p.read_text() == src, f"{backend}/{label}: restore failed"
    rows.append(r)
    print(f"  {backend:7s} {label:28s} new_failed={r['new']['failed']:3d} old_failed={r['pre_existing']['failed']:3d}")
facts["mutations"] = rows

CELLS = {
    "isaac": {"fps": 230, "posture": 241, "cameras": 249, "rollout rate": 256},
    "newton": {"fps": 161, "posture": 172, "cameras": 180, "rollout rate": 187},
}
cov = {}
for arm, jf in (("before", f"/tmp/cov-before-{RUN}.json"), ("after", f"/tmp/cov-after-{RUN}.json")):
    data = json.load(open(jf))
    cov[arm] = {}
    for backend, cells in CELLS.items():
        miss = set(data["files"][f"strands_robots/simulation/{backend}/recording.py"]["missing_lines"])
        cov[arm][backend] = {name: (line not in miss) for name, line in cells.items()}
facts["driven"] = cov
facts["cells"] = CELLS

facts["file_coverage"] = {}
for arm, jf in (("pristine", f"/tmp/cov-{RUN}.json"), ("branch", f"/tmp/cov-final-{RUN}.json")):
    facts["file_coverage"][arm] = {}
    d = json.load(open(jf))
    for backend in ("isaac", "newton"):
        s = d["files"][f"strands_robots/simulation/{backend}/recording.py"]
        facts["file_coverage"][arm][backend] = {
            "miss": len(s["missing_lines"]), "pct": round(s["summary"]["percent_covered"], 1)
        }

prog = (
    "import json,sys\n"
    "sys.path.insert(0,'tests')\n"
    "from tests.simulation.test_recording_preflight_refusals_across_backends import _isaac_engine,_newton_engine\n"
    "out={}\n"
    "for n,f in (('isaac',_isaac_engine),('newton',_newton_engine)):\n"
    "    e=f()\n"
    "    out[n]={'rates':e._active_rollout_rates(),\n"
    "            'guard':e._validate_recording_start_rate(30,'start_recording')}\n"
    "print(json.dumps(out))\n"
)
r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                   env={**os.environ, "HF_HUB_OFFLINE": "1"}, cwd=str(ROOT))
facts["is_rate_unreachable"] = json.loads(r.stdout.strip().splitlines()[-1])
print("rate cell:", facts["is_rate_unreachable"])

pathlib.Path(f"/tmp/facts-{RUN}.json").write_text(json.dumps(facts, indent=2))
print(f"\nwrote /tmp/facts-{RUN}.json")
