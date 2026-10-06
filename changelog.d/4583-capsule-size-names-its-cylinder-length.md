### Fixed: add_object names a capsule's size[2] as its cylinder length on every surface

The agent tool schema and the size-layout table called a capsule's `size[2]` its
full height, while MuJoCo builds it as the cylindrical section and the two caps
add one diameter on top: `size=[0.04, 0, 0.20]` stands 0.24 m. The schema, the
layout quoted in refusals and the `_normalize_size` docstring now state the
`size[2] + size[0]` rule the method docstring and docs page already gave, and the
success text names the split (`0.2 m cylinder + 0.04 m of end caps`). The built
geometry is unchanged.
