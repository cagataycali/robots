#!/usr/bin/env python3
"""Reproduce: `strands-robots doctor` says PASS on a lerobot below the pinned floor.

pyproject.toml pins:
    lerobot[feetech,dataset]>=0.6.1,<0.7.0
    lerobot[molmoact2]>=0.6.1,<0.7.0
    lerobot[smolvla]>=0.6.1,<0.7.0

But strands_robots/doctor.py:302-325 check_lerobot() only checks import-ability
via lerobot_install_error() (utils.py:151), then reports _pass(f"lerobot {ver}")
with NO version comparison against the floor.

Result: `pip install strands-robots` on a host that already has lerobot 0.5.1
(e.g. from strands-wfm) silently gets an unsatisfied pin. `doctor` says "All
checks passed. Ready to use strands-robots." Then first `stream_dataset(...)`
crashes with a TypeError naming a kwarg the user never passed:

    TypeError: StreamingLeRobotDataset.__init__() got an unexpected
    keyword argument 'return_uint8'

`return_uint8` and `repo_type` are wrapper defaults injected at
strands_robots/streaming_dataset.py:275,330. lerobot 0.6.1 accepts them; 0.5.1
does not. There is no runtime version guard anywhere, despite pyproject.toml
line 344-346 stating "the same version for the runtime guard that protects an
environment with a pre-existing older lerobot."
"""
import subprocess, sys
from importlib.metadata import version, PackageNotFoundError

def show(cmd, *, timeout=30):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr

# 1) Show the version + the pin
try:
    v = version("lerobot")
except PackageNotFoundError:
    print("lerobot is NOT installed - this repro needs 0.5.x (below the pinned floor 0.6.1).")
    sys.exit(0)

print(f"Installed lerobot: {v}")
print("Pinned floor (pyproject.toml [project.optional-dependencies].lerobot): >=0.6.1,<0.7.0")

# 2) doctor treats this as PASS
rc, out, err = show(["strands-robots", "doctor"])
for line in out.splitlines():
    if "lerobot" in line or "All checks passed" in line:
        print(f"  doctor> {line}")

# 3) The first documented user call blows up
print()
print("Now try the literal docs/learn/data/stream-and-sync.md example (line 10):")
print('    reader = stream_dataset("you/so101_reach", episodes=[0, 1], buffer_size=1)')
try:
    from strands_robots.streaming_dataset import stream_dataset
    stream_dataset("you/so101_reach", episodes=[0, 1], buffer_size=1)
except TypeError as e:
    print(f"  -> TypeError: {e}")
    print()
    print("The error names return_uint8 - a kwarg the user never passed. It is a")
    print("wrapper default (streaming_dataset.py:214 and 275/330), and lerobot 0.5.x")
    print("does not accept it. doctor should have refused the environment.")
    sys.exit(1)
