### Fixed: `load_scene` says what it dropped

`load_scene` replaces the live world, discarding every registered robot,
object and camera. Its result now names them (`REPLACED the live world:
dropped robot(s) ['so101'] …`) and spells the `add_robot(name=…,
data_config=…)` / `add_object` / `add_camera` calls that put them back into
the loaded scene, with a json block of the dropped names. Previously the loss
surfaced only later as "No robots registered".
