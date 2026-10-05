"""
bugbash repro: docs/learn/mesh/bridges.md:35 omits `lidar` and `map` from the
LAN-only list, but both are real published topics (sensors.py:609/614 and
sensors.py:700) that are NOT in `DEFAULT_BRIDGE_SUFFIXES` -- i.e. they are
LAN-only by default. A fleet owner who reads the doc paragraph literally and
builds a cloud dashboard listing "what reaches AWS IoT Core" will omit lidar
and map and therefore not realise those topics stay on the LAN.

Deterministic verification only -- no network, no simulator -- so this
reproduces identically on CI, Thor, laptop, inside a devcontainer.

Expected (what docs claim):
    LAN-only suffixes = {state, pose, imu, odom, camera, input, hand, stream}

Actual (what code enforces):
    LAN-only suffixes = {state, pose, imu, odom, camera, input, hand, stream,
                         lidar, map}

Reproduces: v0.5.2 (installed), upstream HEAD 52a4530 (2026-11-25).
"""
from __future__ import annotations

import os
import re
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _published_topic_suffixes() -> set[str]:
    """Scan sensors.py + core.py for every `publish(f"strands/{...}/<suffix>...")`
    call and return the first path segment of each unique suffix.
    """
    first_segments: set[str] = set()
    # These three files fan every non-RPC topic out onto the mesh.
    for name in (
        "strands_robots/mesh/sensors.py",
        "strands_robots/mesh/core.py",
        "strands_robots/mesh/input.py",
    ):
        src = (REPO / name).read_text(encoding="utf-8")
        for m in re.finditer(r'publish\(\s*f?"strands/\{[^}]+\}/([^"\s/]+)', src):
            first_segments.add(m.group(1))
    # Drop RPC-shape and bridged-by-design ones.
    rpc = {"cmd", "response", "broadcast", "presence", "health", "safety"}
    return first_segments - rpc


def main() -> None:
    # 1. Guarantee a stock environment (no user-set bridge filter).
    os.environ.pop("STRANDS_MESH_BRIDGE_TOPICS", None)
    os.environ.pop("STRANDS_MESH_BRIDGE_TOPICS_PREFIX", None)

    # 2. Resolve what the code bridges by default.
    from strands_robots.mesh.transport.bridge_transport import (
        DEFAULT_BRIDGE_SUFFIXES,
        _resolve_bridge_filter,
    )
    bridged = _resolve_bridge_filter()
    assert bridged == DEFAULT_BRIDGE_SUFFIXES, "stock env must give the default set"

    # 3. Discover what topics the mesh actually publishes.
    published = _published_topic_suffixes()
    # `input` is published on strands/<peer>/input/<device> but the publisher is
    # built via a `topic` property (mesh/input.py:298), not a `publish(f"...")`
    # literal, so it won't be in the regex scan. Add it by name.
    published.add("input")
    assert {"state", "pose", "imu", "odom", "camera", "input", "hand", "stream",
            "lidar", "map"}.issubset(published), (
        f"regex drift: published={sorted(published)}"
    )

    # 4. Compute the real LAN-only set.
    lan_only_in_code = {s for s in published if s not in bridged}

    # 5. Parse the doc paragraph.
    doc_src = (REPO / "docs/learn/mesh/bridges.md").read_text(encoding="utf-8")
    # The claim is on one line, around line 35 at HEAD.
    for ln, line in enumerate(doc_src.splitlines(), start=1):
        if "LAN-only:" in line:
            doc_line_no = ln
            doc_line = line
            break
    else:
        raise SystemExit("docs/learn/mesh/bridges.md no longer carries the LAN-only sentence")

    # Only parse the backticked words AFTER the "LAN-only:" marker.
    doc_lan_only_section = doc_line.split("LAN-only:", 1)[1]
    doc_lan_only = set(re.findall(r"`([a-z]+)`", doc_lan_only_section))

    print(f"docs/learn/mesh/bridges.md:{doc_line_no}")
    print(f"  doc says LAN-only: {sorted(doc_lan_only)}")
    print(f"  code enforces     : {sorted(lan_only_in_code)}")

    missing_from_doc = sorted(lan_only_in_code - doc_lan_only)
    print(f"\nMISSING from doc paragraph: {missing_from_doc}")

    if missing_from_doc:
        print(f"\n✗ DEFECT PRESENT: docs/learn/mesh/bridges.md:{doc_line_no}")
        print(f"  under-specifies the LAN-only set by {len(missing_from_doc)} "
              f"entry/entries: {missing_from_doc}.")
        print(f"  A reader building a cloud audit view from this paragraph will")
        print(f"  miss these live telemetry categories that production fleets DO")
        print(f"  emit when sensors are present.")
        raise SystemExit(1)
    else:
        print(f"\n✓ docs/learn/mesh/bridges.md:{doc_line_no} now names every LAN-only topic")

    # 6. Code evidence that lidar and map actually publish on the mesh.
    sensors = (REPO / "strands_robots/mesh/sensors.py").read_text(encoding="utf-8")
    for key, needle in (
        ("lidar/summary", 'f"strands/{self.peer_id}/lidar/summary"'),
        ("lidar/state",   'f"strands/{self.peer_id}/lidar/state"'),
        ("map/info",      'f"strands/{self.peer_id}/map/info"'),
    ):
        assert needle in sensors, f"sensors.py no longer publishes {key}"
    print("\nAll three LAN-only publishes still present at:")
    print("  sensors.py:609  -> strands/{peer}/lidar/summary")
    print("  sensors.py:614  -> strands/{peer}/lidar/state")
    print("  sensors.py:700  -> strands/{peer}/map/info")

    # 7. Setting STRANDS_MESH_BRIDGE_TOPICS to the doc's list would silently
    #    lose lidar/map coverage in a would-be dashboard built from the doc.
    os.environ["STRANDS_MESH_BRIDGE_TOPICS"] = ",".join(sorted(doc_lan_only))
    _ = _resolve_bridge_filter()  # just a sanity call; no assertion.


if __name__ == "__main__":
    main()
