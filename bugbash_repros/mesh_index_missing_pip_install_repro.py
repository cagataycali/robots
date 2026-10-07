#!/usr/bin/env python3
"""Repro: docs/learn/mesh/index.md fence silently fails on a fresh `pip install strands-robots`.

A reader lands on docs/learn/mesh/index.md as the entry page for the mesh
ladder (linked from docs/start/install.md "Add [...] mesh when a second machine
does"). The page's first fence runs under a `Robot("so101", mesh=True)` call
and claims the two peers see each other:

    print(a.mesh.alive, sorted(p["peer_id"] for p in a.mesh.peers))
    # True ['arm-a__so101', 'arm-b', 'arm-b__so101']

What actually happens on a bare `pip install strands-robots` -- the install
line the sibling `docs/start/install.md` shows one page earlier (which pulls
only `strands-agents`, `numpy`, `opencv-python-headless`, `Pillow`; mesh is
opt-in through the `[mesh]` extra, pulling `eclipse-zenoh`) -- is reproduced
below by shadowing `zenoh` out of the import graph.

The runtime emits ONE loud WARNING naming the install line, but the Python
block the page shows keeps running: `mesh.alive` is False, `peers` is empty,
and `send` answers `status="error"` -- byte-identical to a healthy mesh that
just has no partner yet.

Expected (what the fence promises): the two peers appear and send() works.
Actual (what the user gets): empty peer list, mesh not running, send errors.

Fix: carry the one-line install snippet the sibling page `bridges.md:10`
already has, so the first instruction the user reads for the mesh page is to
install the extra it needs.
"""

from __future__ import annotations

import logging
import os
import sys


def _hide_zenoh() -> None:
    """Make `import zenoh` fail, the way a bare `pip install strands-robots` env would."""

    class _ShadowFinder:
        def find_spec(self, name, path, target=None):  # noqa: ANN001 - importlib signature
            if name == "zenoh" or name.startswith("zenoh."):
                raise ImportError(f"shadowed-for-repro: {name}")
            return None

    sys.meta_path.insert(0, _ShadowFinder())
    for key in [k for k in sys.modules if k == "zenoh" or k.startswith("zenoh.")]:
        del sys.modules[key]


def main() -> int:
    _hide_zenoh()
    os.environ["STRANDS_MESH_LOCAL_DEV"] = "true"
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # Copy of the first fence from docs/learn/mesh/index.md (lines 12-28 at
    # commit 8a99e0555), two sims instead of two processes so one script shows it.
    from strands_robots import Robot

    a = Robot("so101", mesh=True, peer_id="arm-a")
    _ = Robot("so101", mesh=True, peer_id="arm-b", tool_name="so101_b")

    print(f"a.mesh.alive = {a.mesh.alive!r}")
    print(f"a.mesh.peers = {a.mesh.peers!r}")
    reply = a.mesh.send("arm-b", {"action": "status"}, timeout=1.0)
    print(f"a.mesh.send -> {reply!r}")

    # The docstring the page prints under its own line is `(True, [...peers...])`.
    # We land on `False, []`, and the only breadcrumb is a WARNING that scrolls past.
    ok = a.mesh.alive is True and a.mesh.peers and reply.get("status") != "error"
    print()
    print("docs/learn/mesh/index.md fence runs as advertised:", bool(ok))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
