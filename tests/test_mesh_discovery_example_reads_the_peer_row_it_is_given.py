"""``examples/04_mesh_peer_discovery.py`` reads the peer row the registry hands it.

Observed: the example printed ``type=?`` for every peer. ``PeerInfo.to_dict()``
serialises the peer's kind under ``"type"``; the example read ``"peer_type"``,
a key no row carries. It also queried the registry straight after ``Robot()``
returned - before any other process's heartbeat (``HEARTBEAT_HZ``) could land -
so it could only ever list this process's own robots, and it listed them as
discovered peers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from strands_robots.mesh.session import HEARTBEAT_HZ, PeerInfo

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "04_mesh_peer_discovery.py"


def _string_keys_read_from_peer_rows(source: str) -> set[str]:
    """Every literal key the example reads off a ``peer`` dict."""
    keys: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "peer"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            keys.add(str(node.args[0].value))
    return keys


def test_every_key_the_example_reads_is_one_the_registry_writes():
    row = PeerInfo(peer_id="p", peer_type="sim", hostname="h", last_seen_mono=0.0, caps={}).to_dict()
    read = _string_keys_read_from_peer_rows(_EXAMPLE.read_text())
    assert read, "the example reads no peer keys - the scan is broken"
    assert read <= set(row), f"example reads keys no peer row carries: {sorted(read - set(row))}"
    assert "type" in read


def test_the_example_waits_for_a_heartbeat_before_reading_the_registry():
    source = _EXAMPLE.read_text()
    sleep = re.search(r"time\.sleep\(\s*(\d+)\s*/\s*HEARTBEAT_HZ\s*\)", source)
    assert sleep, "the example must wait a heartbeat-scaled interval before get_peers()"
    assert int(sleep.group(1)) / HEARTBEAT_HZ >= 1 / HEARTBEAT_HZ
    assert source.index("time.sleep(") < source.index("peers = get_peers()")


def test_the_example_separates_its_own_robots_from_discovered_peers():
    source = _EXAMPLE.read_text()
    assert "not in local" in source and "in local]" in source
