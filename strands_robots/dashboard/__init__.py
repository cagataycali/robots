"""Strands Robots Dashboard - fleet cockpit for the robot mesh.

Start with::

    strands-robots dashboard              # native
    strands-robots dashboard --port 9000

The dashboard is a mesh *peer*, not a hub: it joins the same Zenoh session
every ``Robot(mesh=True)`` / ``Robot().run()`` uses and renders whatever it
discovers - presence, joint state, cameras (hardware AND sim), VLA step
streams, and safety events.
"""

__all__ = ["main"]


def main() -> None:
    from strands_robots.dashboard.cli import main as cli_main

    cli_main()
