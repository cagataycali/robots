### Fixed: Isaac mesh objects compose as meshes

A mesh added with `add_object(shape="mesh")` or a `load_scene` mesh visual composed as an empty `Xform`, because the placeholder prim's type overrode the referenced `Mesh`: it rendered nothing and a dynamic one fell through the floor. The placeholder's type is now cleared, so the prim composes as the mesh it references.
