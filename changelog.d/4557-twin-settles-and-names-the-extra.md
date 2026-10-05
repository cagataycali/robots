### Fixed: the dashboard's Twin button says why a twin did not start, and offers the install

`POST /api/robots/{peer}/twin` returned `200` with a pid the moment the child was
forked, so a twin that died on `import mujoco` a second later looked like a click
that did nothing. The twin route now takes the same path as `POST /api/devices/spawn`:
a spawn this environment cannot run is refused with a `412` naming the extra
(`sim-mujoco`), a child that dies inside the settle window answers `status: failed`
with its reason and `missing_extra`, and both land in the activity trail. The
spawn preflight now checks a `mode="sim"` spawn for MuJoCo, so the Devices sheet
refuses a sim spawn the same way. The robot card and detail view show the reason
and an install button for the named extra.
