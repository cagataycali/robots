### Fixed: Isaac loads a robot whose MJCF attaches another model (`lekiwi`)

MuJoCo composes a `<model>` asset used by `<attach>` as its own spec, so the
attached model keeps its own `<compiler meshdir>`. The Isaac MJCF importer resolves
every mesh against the entry model instead, so `add_robot("lekiwi")` (a base that
attaches `../so_arm100/so_arm100.xml`) failed with `Mesh Base file
.../lekiwi/assets/Base.stl` although MuJoCo loads it. A model that attaches another
now reaches the importer as one flattened file (`MjSpec.to_xml` of the composed
model, every mesh and texture path absolute, checked to compile to the same model);
any other description reaches it untouched. `lekiwi` loads on Isaac Sim 6.1 with
its 9 joints.
