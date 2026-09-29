### Fixed: `Robot()` refuses a non-string name with the `ValueError` it promises

`Robot(None)`, `Robot(123)` and `Robot(["so101"])` escaped as an
`AttributeError` from the name fold and `Robot(b"so101")` as a `TypeError` from
a regex. The factory now refuses a non-string name at the door with the wording
the empty string gets, naming the type it received, `list_robots()` and
`urdf_path=`. Closes #4150.
