"""No dashboard bookkeeping key may cross into a child's argv or a generated file.

camera_liveness stamps ``device_name`` into a camera config so the dashboard remembers WHICH physical
camera an index was. hardware_robot refuses an unknown camera option by name, and that refusal kills
every camera on the arm - so the key is legal inside the dashboard, legal on its own API (the
reconfigure editor round-trips it), and fatal the moment it reaches something that builds a real robot.

That asymmetry has already produced one shipped bug: the deploy snippet rendered the camera mapping
verbatim, so every snippet generated for a stamped arm was a file that died at connect on the edge
device (fixed 2026-08-22, b5e07409). The spawn path had stripped it correctly all along, which is
exactly why the miss was invisible - one owner of the rule, two boundaries, only one calling it.

This is a SOURCE CENSUS, not a behaviour test: the two dangerous shapes are a subprocess payload
(``json.dumps({... "cameras": ...})``) and generated Python (``cameras={_fmt(...)}``). Both are
grepped and both must pass through ``without_annotations``. A sink of some THIRD shape is not covered
and this file says so rather than implying a completeness it cannot prove - the count it reports is
the count it checked.
"""
from __future__ import annotations

import re
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "strands_robots" / "dashboard"

#: A line that hands `cameras` to a child process or writes it into generated code.
_SINK_PATTERNS = (
    re.compile(r"dumps\(.*[\"']cameras[\"']"),      # subprocess payload
    re.compile(r"cameras=\{?_fmt\("),                # generated python
)


def _sink_lines() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for path in sorted(DASHBOARD.glob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if any(p.search(line) for p in _SINK_PATTERNS):
                found.append((path.name, n, line.strip()))
    return found


def test_every_outward_camera_sink_strips_the_annotation():
    sinks = _sink_lines()
    # If this is zero the census has stopped measuring anything (a rename, a refactor) and a green
    # here would be a false one - the same "narrowed to nothing" failure the audit runner reports.
    assert sinks, "no camera sink found at all: this guard has stopped measuring, fix the patterns"
    leaking = [(f, n, t) for f, n, t in sinks if "without_annotations" not in t]
    assert not leaking, (
        f"{len(leaking)} of {len(sinks)} camera sink(s) hand a config outward WITHOUT stripping the "
        f"dashboard's own device_name; hardware_robot refuses it and every camera on the arm dies: "
        + "; ".join(f"{f}:{n} {t}" for f, n, t in leaking)
    )


def test_the_stripper_still_removes_what_the_child_refuses():
    """The census above is worthless if the function it insists on calling stops working."""
    from strands_robots.dashboard.camera_liveness import ANNOTATION_KEYS, without_annotations

    assert ANNOTATION_KEYS, "an empty annotation set would make every strip a no-op"
    stamped = {"main": {"index_or_path": 0, "fps": 30, **{k: "x" for k in ANNOTATION_KEYS}}}
    out = without_annotations(stamped)
    assert out is not None
    for key in ANNOTATION_KEYS:
        assert key not in out["main"]
    assert out["main"] == {"index_or_path": 0, "fps": 30}
    # An unstamped mapping is returned UNCHANGED (identity), so the common path allocates nothing.
    plain = {"main": {"index_or_path": 0}}
    assert without_annotations(plain) is plain
