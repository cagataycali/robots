### Fixed: a patch_scene_mjcf op key outside its vocabulary is rejected, not defaulted

Every field of every structured op was read with a fallback default, and no op
checked for keys it does not read. A key the op did not recognise therefore left
that default in place and the patch reported success:

```python
sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate", "position": [0.4, 0, 0.9]}])
# before: status=success "1 op(s) applied" - and the crate moved to [0, 0, 0]
```

`pos` defaults to the origin, `quat` to identity, `type` to `"box"` and `parent`
to the worldbody, so the same hole covered six ops: a misspelled `pos` teleported
a body to the world origin (or spawned it there), a misspelled `quat` reset the
orientation to identity, `add_geom` with `shape=`/`color=` compiled a grey box
where a coloured sphere was asked for, `add_site` with a misspelled `pos` placed
the site at the body origin, and `add_body` with `parent_body=` re-parented the
new body to the worldbody instead of the intended parent.

Each op now declares the keys it reads, and anything else is refused with the op
name, the unrecognised key, a close match where one exists, and the accepted
list. The batch stays atomic, so a rejected key leaves the scene exactly as it
was rather than half-patched.
