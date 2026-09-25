"""``examples/04_mesh_peer_discovery.py`` reads the peer row the registry hands it.

Observed: the example printed ``type=?`` for every peer. ``PeerInfo.to_dict()``
serialises the peer's kind under ``"type"``; the example read ``"peer_type"``,
a key no row carries. It also queried the registry straight after ``Robot()``
returned - before any other process's heartbeat (``HEARTBEAT_HZ``) could land -
so it could only ever list this process's own robots, and it listed them as
discovered peers.

Its own closing hint is to start the script in a second terminal. It handed every
copy one hardcoded ``peer_id``, and the registry files a record under that id, so
the second copy overwrote the first's row and each read the other as itself:
measured, two terminals both printed ``Discovered mesh peers: 0`` beside the hint
telling them to do what they had just done, where the same example with distinct
ids printed the remote arm. The identity a second copy claims must therefore be
one this process cannot share.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
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


def _identity_expression() -> str:
    """The source of the ``peer_id`` the example hands ``Robot()``.

    A name is resolved to the module-level expression assigned to it, so the
    identity is read whether it is written at the call or derived above it.
    """
    tree = ast.parse(_EXAMPLE.read_text())
    given: ast.expr | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Robot":
            for kw in node.keywords:
                if kw.arg == "peer_id":
                    given = kw.value
    assert given is not None, "the example does not pass peer_id to Robot() - the scan is broken"
    if not isinstance(given, ast.Name):
        return ast.unparse(given)
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == given.id for target in statement.targets
        ):
            return ast.unparse(statement.value)
    raise AssertionError(f"peer_id is the name {given.id!r}, assigned nowhere at module level")


def _evaluate(times: int, per_process: int) -> list[str]:
    """Evaluate the identity expression *per_process* times in *times* processes."""
    expression = _identity_expression()
    program = f"import os, sys\nfor _ in range({per_process}): print({expression})"
    out: list[str] = []
    for _ in range(times):
        done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True)
        out.extend(done.stdout.split())
    return out


def test_a_second_copy_of_the_example_claims_a_different_identity():
    ids = _evaluate(times=3, per_process=1)
    assert len(set(ids)) == len(ids), (
        f"every copy of the example joins the mesh as {ids[0]!r}: two of them overwrite "
        "one peer record and each reads the other as itself"
    )


def test_the_identity_does_not_change_under_the_process_that_claimed_it():
    ids = _evaluate(times=1, per_process=3)
    assert len(set(ids)) == 1, f"the identity is not stable within one process: {ids}"
    assert all(ids), "the identity is empty"
