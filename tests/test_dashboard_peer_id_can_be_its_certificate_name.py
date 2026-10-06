"""The dashboard's mesh peer id can be pinned to the name its certificate speaks for.

``MeshBridge`` used to pick ``dashboard-<host>-<4 hex>`` at every start. On a
mesh that requires signed identity a peer is heard only under a name its
certificate's common name speaks for, so a random id would have every robot
drop the dashboard's presence and refuse its motion commands.
``STRANDS_DASHBOARD_PEER_ID`` names it; an explicit constructor argument still
wins, and with neither the old shape is kept.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs the [dashboard] extra")

from strands_robots.dashboard.mesh_bridge import PEER_ID_ENV, MeshBridge  # noqa: E402
from strands_robots.mesh.wire_identity import cn_speaks_for  # noqa: E402


def test_the_variable_names_the_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PEER_ID_ENV, "lab-console")

    bridge = MeshBridge()

    assert bridge.peer_id == "lab-console"
    assert cn_speaks_for("lab-console", bridge.peer_id)


def test_an_explicit_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PEER_ID_ENV, "lab-console")

    assert MeshBridge(peer_id="other").peer_id == "other"


def test_unset_keeps_the_generated_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PEER_ID_ENV, raising=False)

    assert MeshBridge().peer_id.startswith("dashboard-")
