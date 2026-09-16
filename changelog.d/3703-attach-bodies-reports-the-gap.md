### Fixed: `attach_bodies` says whether the bodies touch

`attach_bodies` welded any two bodies at their current offset and said only
"not a physical grasp". An agent running the README quickstart could not reach
the cube, closed the gripper centimetres away, attached the cube and reported a
successful pick - the cube was riding along 7 cm below the fingers. The tool
now measures the closest surface distance between the parent's subtree and the
child (`mj_geomDistance`); when they do not touch the text says so in
centimetres ("floats rigidly at that offset - a weld, not a pick") and the new
json payload carries `gap_m` and `touching`. Attaching at a distance is still
allowed - mounting a sensor is a real use.
