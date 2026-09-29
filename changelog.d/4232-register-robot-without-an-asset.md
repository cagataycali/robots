### Added: `register_robot` registers a robot with no simulation asset

`model_xml` is now optional, so the public writer can produce the same
hardware-only entry the package registry already ships - a remote robot, or a
sensor-only one with no MJCF/URDF. Without `model_xml` the entry is stored with
no `asset` block, and it must declare `hardware` with a non-empty `lerobot_type`
or `driver` `"strands"`, so the entry states how it is driven for real (a
`strands` driver still has to be registered with `register_native_driver`):

```python
register_robot("drone", category="aerial", hardware={"driver": "strands"})
```

`scene_xml`, `asset_dir` and `robot_descriptions_module` only describe an asset
and are refused without `model_xml`. `Robot("drone", mode="sim")` reports
the robot as real-hardware only. Registration with `model_xml` is unchanged,
except that a blank `model_xml` is a `ValueError` and a non-`str` one (a `Path`,
an `int`) or a non-`dict` `hardware` is a `TypeError` naming the type. Blank
checks use `str.strip`, so a `str` subclass that overrides `strip` cannot hide a
blank value.
