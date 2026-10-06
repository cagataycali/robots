### Fixed: `list_urdf_only()` lists only the URDF robots that build

`list_urdf_only()` returned every URDF-only description, including the ones
`urdf_robots.json` records as not building (`has_sim` false, e.g. `eve_r3`),
so a name copied from it could fail in `Robot(name)`. It now leaves those out;
`list_urdf_only(include_refused=True)` returns the full set, which
`list_robots()` and `scripts/build_urdf_registry.py` use. `is_urdf_only()` and
`Robot(name)` still recognise a refused name and report its refusal.
