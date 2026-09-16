### Added: a 3D twin in the browser, drawn from the compiled model

`/api/sim/{id}/scene` describes the geoms, meshes and cameras of the session's
`MjModel`; `/api/sim/{id}/mesh/{i}` serves each compiled mesh as bytes read
straight out of `mesh_vert`/`mesh_face` - no file is opened, so there is no
path to contain. With `?poses=1` the telemetry socket follows every JSON
snapshot with one binary frame of `geom_xpos|geom_xmat` rows, and
`static/twin.js` (three.js r170, vendored under `static/vendor/` with its MIT
notice so the page works on a LAN with no internet) sets each object's matrix
from it. No physics runs in the browser and no MJCF is parsed there: the twin
shows exactly what MuJoCo computed, for every robot the engine can load.

The frame's row width is published as `pose_row_floats` and owned by
`scene.POSE_ROW_FLOATS`: the packer emits that width and `twin.js` strides by
the value the scene sent it, so neither restates it.
