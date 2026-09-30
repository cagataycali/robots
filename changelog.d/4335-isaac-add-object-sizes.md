### Fixed: an Isaac add_object size builds the object it describes, or is refused by name

The published tool schema, and the MuJoCo backend, spell a cylinder or capsule
`size` as `[diameter, unused, full height]`; Isaac read the same list as
`[radius, height]`. `add_object(shape="cylinder", size=[0.04, 0, 0.06])` built a
zero-height collider that fell through the ground to z = -19.9 m in two seconds
under a success envelope, and `[0.04, 0.04, 0.06]` built a cylinder 4 cm tall
with a 4 cm radius instead of 6 cm tall with a 2 cm radius. A box with a zero
extent failed deep in USD ("Non-positive determinant ... in rotation matrix").

A three-component cylinder/capsule size now uses the `[diameter, unused, full
height]` layout, so one size builds the same object on both backends; the
two-component `[radius, height]` form keeps its meaning. Every extent a shape
consumes must be greater than zero, and a size that breaks this is refused,
naming the component, before any prim is created.
