### Fixed: `download_assets(robots="open_duck_mini")` succeeds - the registry names the model file the upstream tree has

The `open_duck_mini` entry declared `open_duck_mini_v2.xml`, which the upstream tree
(`apirrone/Open_Duck_Mini`, `mini_bdx/robots/open_duck_mini_v2`) does not contain - it
holds `robot.xml`, `robot_motors.xml`, `scene.xml` and `scene_position.xml` - so every
download reported "Failed: 1 ... fetched tree has no open_duck_mini_v2.xml" and left
the tree on disk, and the next `add_robot("open_duck_mini")` loaded it anyway from
`scene.xml`. The entry (and the docs viewer's copy) now declares `robot_motors.xml`,
the model `scene.xml` includes; the same download reports "Downloaded: 1, Failed: 0".
