"""Strands Robots Dashboard — a real-time web UI for the Zenoh robot mesh.

The dashboard joins the mesh as an observer peer (``peer_type="dashboard"``)
and fans mesh traffic out to browser clients over WebSocket. It reuses the
existing :class:`strands_robots.mesh.Mesh` machinery, so it inherits the
mesh's mTLS transport, ACL gating, command validation, and audit logging —
the dashboard never touches raw Zenoh.

Run it::

    strands-robots dashboard --port 7860

or from Python::

    from strands_robots.dashboard import start_dashboard
    start_dashboard(port=7860)
"""

from strands_robots.dashboard.server import start_dashboard

__all__ = ["start_dashboard"]
