"""Measure sync_dataset_to_bucket against the real `hf` CLI of several releases.

Only the version string the gate reads is emulated; the CLI invoked is the
genuine `hf` from a shadow venv of that release, so each row is the outcome a
caller on that release really gets.
"""
import json, pathlib, subprocess, sys, tempfile
import huggingface_hub
import strands_robots.dataset_recorder as dr

TREE = str(pathlib.Path(dr.__file__).parents[1])
SHADOWS = {"1.0.0": "/tmp/shadow1.0.0/bin/hf",
           "1.4.1": "/tmp/shadow1.4.1/bin/hf",
           "1.5.0": "/tmp/shadow1.5.0/bin/hf"}
BUCKET = "cagataydev/floor-artifact"

root = pathlib.Path(tempfile.mkdtemp()) / "cube_pick"
(root / "meta").mkdir(parents=True)
(root / "meta" / "info.json").write_text('{"fps": 30}')

out = {"tree": TREE, "rows": {}}
real = dr._hf_executable
for ver, exe in SHADOWS.items():
    huggingface_hub.__version__ = ver
    dr._hf_executable = lambda _e=exe: _e
    # real CLI capability probe (no network, no mutation)
    rc = subprocess.run([exe, "buckets", "--help"], capture_output=True).returncode
    gate = dr._huggingface_hub_version_error()
    res = dr.sync_dataset_to_bucket(root=root, bucket=BUCKET, create=True)
    out["rows"][ver] = {
        "cli_buckets_help_rc": rc,
        "gate_refused": gate is not None,
        "status": res["status"],
        "message": (res.get("message") or ("synced -> " + res.get("bucket_uri", "")))[:200],
    }
dr._hf_executable = real
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("wrote", sys.argv[1], "tree:", TREE)
