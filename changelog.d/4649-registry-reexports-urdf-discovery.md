### Fixed: the URDF discovery probes import from `strands_robots.registry` like their MJCF siblings

`is_urdf_discoverable`, `list_urdf_discoverable`, `discover_urdf_path` and
`urdf_descriptions_module` are now in `strands_robots.registry.__all__` next to
`is_discoverable`, `list_discoverable`, `discover_robot` and
`descriptions_module`, so `from strands_robots.registry import
list_urdf_discoverable` no longer raises `ImportError`. The registry API page
lists them.
