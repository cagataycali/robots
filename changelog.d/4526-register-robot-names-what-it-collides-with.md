### Fixed: `register_robot()` names what it collides with

Refusing a taken name used to call every entry a "built-in robot" and print
`0 joints` when no count was recorded. A name from the `robot_descriptions`
long tail (`list_urdf_only()`) is now called an auto-discovered URDF, the joint
count appears only when the registry has one, and an entry that does not build
(`eve_r3`) shows its recorded reason and points at `overwrite=True` as the way
to supply a working model.
