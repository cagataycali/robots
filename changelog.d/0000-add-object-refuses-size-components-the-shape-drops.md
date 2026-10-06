### Fixed: `add_object` refuses a `size` component the shape would silently drop

A sphere given `size=[0.05, 0.10, 0.20]` compiled as a 5 cm ball and reported
success; a cylinder's "unused" middle component and a plane's third were
dropped the same way. A component the shape does not consume may now be `0` or
repeat the extent it mirrors (`[d, d, d]` on a sphere, the 5 cm default, and
`[diameter, 0, height]` keep working); any other value is refused with the
index it names, and a sphere refusal points at `shape="ellipsoid"`.
