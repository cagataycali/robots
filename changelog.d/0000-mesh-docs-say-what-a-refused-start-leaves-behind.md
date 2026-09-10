### Documentation
- `docs/mesh.md` no longer says `robot.mesh` becomes `None` when the mesh fails: it is `None` only when switched off, and a `Mesh` with `alive=False` when it did not start - the observable `docs/troubleshooting.md` and the fleet examples already use.
