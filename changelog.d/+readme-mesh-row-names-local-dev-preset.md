### Changed

- README L97 "What you get" Mesh row now names the single-machine preset
  `STRANDS_MESH_LOCAL_DEV=true` next to `Robot(mesh=True)`. The previous phrasing
  promised the mesh worked out of the box but the permissive-ACL gate refuses
  `Mesh.start()` under the default `STRANDS_MESH_AUTH_MODE=mtls` with no ACL file
  configured. A reader following the row literally got
  `{'status': 'error', 'error': 'mesh not running'}` from the exact
  `robot.mesh.tell(...)` call the row names. The one-liner the mesh index page
  (`docs/learn/mesh/index.md:9`) uses to open its own fence is now named on the
  README itself.
