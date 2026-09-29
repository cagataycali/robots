### Fixed: `Robot(mesh=True)` without the `[mesh]` extra names the missing dependency

`Mesh.start()` now reports `eclipse-zenoh is not installed ... pip install 'strands-robots[mesh]'` before any posture check, instead of the permissive-ACL refusal banner, which guards no transport when Zenoh is absent. The refusal still fires when Zenoh is installed and the ACL is permissive.
