### Added: the URDF-side discovery functions are re-exported alongside their MJCF siblings

`strands_robots.registry.discovery` exposes four sibling pairs: the cheap
name-to-module lookups (`is_discoverable`/`is_urdf_discoverable`,
`descriptions_module`/`urdf_descriptions_module`), the cheap sorted rosters
(`list_discoverable`/`list_urdf_discoverable`), and the heavy path resolvers
(`discover_robot`/`discover_urdf_path`). Before this change the MJCF side of
each pair was re-exported from `strands_robots.registry.__init__` (and the
two cheap probes further promoted to the top-level `strands_robots` package),
while the URDF side was not re-exported anywhere -- a user following the
sibling pattern `from strands_robots.registry import list_urdf_discoverable`
hit an `ImportError` and had to deep-import from
`strands_robots.registry.discovery`. Internal consumers worked around the
omission the same way (`strands_robots/simulation/newton/simulation.py`,
`scripts/build_urdf_registry.py`).

The four URDF siblings are now re-exported from `strands_robots.registry`;
the two cheap probes (`is_urdf_discoverable`, `list_urdf_discoverable`) are
further promoted to the top-level `strands_robots` package, matching the
shape of their MJCF counterparts. `docs/reference/api/registry.md` now lists
all eight functions under *Discovery through robot_descriptions*. The
Newton-simulation import is updated to the public path.
