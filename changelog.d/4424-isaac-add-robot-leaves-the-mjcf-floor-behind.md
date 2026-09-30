### Fixed: Isaac camera frames of the floor are no longer streak noise

Menagerie's `scene.xml` - what `add_robot` resolves for nearly every robot - puts a
`floor` plane on `<worldbody>`, and the Isaac MJCF importer wrote it under the robot
although `add_robot` converts with `import_scene=False`. It lay exactly on
`create_world()`'s ground plane, and the two coplanar surfaces z-fought, so every RTX
camera frame showed the floor as horizontal-streak noise that MuJoCo frames do not
have (every agent-recorded Isaac dataset carried it). The post-import fix-up now
deactivates every geom the MJCF attaches to `<worldbody>` (the robot's own bodies are
untouched), and the cache key moves so old entries are rebuilt. On so101 the mean
pixel gradient over the floor fell from 4.5 to 0.1.
