"""Measure the joint_limits construction verdict + command consequence in this tree."""

import json
import logging
import pathlib
import sys

import strands_robots
from strands_robots.ros_telemetry import RosTelemetryBase as R

TREE = str(pathlib.Path(strands_robots.__file__).parents[1])
print("TREE:", TREE)

NAN, INF = float("nan"), float("inf")
VALS = [("-1.9", -1.9), ("1.9", 1.9), ("nan", NAN), ("+inf", INF), ("-inf", -INF)]


class Msg:
    def __init__(self, n, p):
        self.name, self.position = n, p


class Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def usable(lo, hi):
    """Contract: a declared range must be able to admit a finite position."""
    finite = lo == lo and hi == hi and abs(lo) != INF and abs(hi) != INF
    return finite and lo <= hi


matrix = {}
for ln, lo in VALS:
    for hn, hi in VALS:
        key = f"{ln}|{hn}"
        try:
            norm = R._validate_joint_limits({"shoulder_pan": (lo, hi)})
            verdict, reason = "accepted", ""
        except ValueError as exc:
            verdict, reason, norm = "refused", str(exc), None
        except OverflowError as exc:
            verdict, reason, norm = "raised", f"OverflowError: {exc}", None
        want = "accepted" if usable(lo, hi) else "refused"
        matrix[key] = {
            "low": ln,
            "high": hn,
            "verdict": verdict,
            "want": want,
            "correct": verdict == want,
            "reason": reason,
        }

# Consequence for the headline case and the valid control: how many of N
# in-range commands reach the arm.
ledger = {}
for label, bounds in [("valid (-1.9, 1.9)", (-1.9, 1.9)), ("nan max (-1.9, nan)", (-1.9, NAN))]:
    cap = Cap()
    log = logging.getLogger("strands_robots.ros_telemetry")
    log.addHandler(cap)
    old = log.level
    log.setLevel(logging.WARNING)
    try:
        norm = R._validate_joint_limits({"shoulder_pan": bounds})
        ctor, err = "accepted", ""
    except (ValueError, OverflowError) as exc:
        ctor, err, norm = "refused", str(exc), None
    applied = 0
    total = 5
    if norm is not None:
        base = R()
        for _ in range(total):
            if base._command_action(Msg(["shoulder_pan"], [0.5]), joint_limits=norm) is not None:
                applied += 1
    log.removeHandler(cap)
    log.setLevel(old)
    ledger[label] = {
        "ctor": ctor,
        "err": err,
        "applied": applied,
        "total": total,
        "log": cap.records[0] if cap.records else "",
    }

# The huge-int class: which exception type escapes.
try:
    R._validate_joint_limits({"j": (10**400, 1.0)})
    huge = {"kind": "accepted", "msg": ""}
except Exception as exc:  # noqa: BLE001 - the exception TYPE is the measurement
    huge = {"kind": type(exc).__name__, "msg": str(exc)[:70] + ("..." if len(str(exc)) > 70 else "")}

out = {"tree": TREE, "matrix": matrix, "ledger": ledger, "huge": huge}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print("wrote", sys.argv[1])
print("  incorrect verdicts:", sum(1 for v in matrix.values() if not v["correct"]), "of", len(matrix))
print("  huge int:", huge["kind"])
