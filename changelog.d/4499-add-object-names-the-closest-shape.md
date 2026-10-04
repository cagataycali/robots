### Fixed: `add_object` names the closest shape for a misspelled `shape=`

`add_object(shape="sfere")` (and `patch_scene_mjcf`'s `add_geom`) now answers
`Unsupported shape 'sfere'. Did you mean 'sphere'? Supported: ...` on MuJoCo,
and the Newton backend gives the same hint, instead of only the list of shapes.
A name close to none of them still gets the plain list.
