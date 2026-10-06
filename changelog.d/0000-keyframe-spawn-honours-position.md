### Fixed: a keyframe spawn lands where `position=` puts it, so the burial warning's advice clears it

`add_robot(keyframe=...)` wrote a floating base's keyframe pose verbatim over
the frame `position=`/`orientation=` attach it under, so a keyframe spawn
ignored `position=` entirely. On `Robot("microduck", urdf_path=scene_rollers.xml,
keyframe="STAND")` the burial warning then named a `position=` that changed
nothing and grew by 20.7 mm each time it was followed. The key's free-joint pose
is now placed inside that frame, at spawn and on every `reset()`, and following
the warning once silences it.
