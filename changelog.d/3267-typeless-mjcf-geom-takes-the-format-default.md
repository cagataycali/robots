### Fixed: a typeless MJCF `<geom>` reads as the sphere the format documents

MJCF gives `<geom>` a default type of `sphere`, so `<geom size="0.03"/>` is a
30 mm ball. The Isaac description loader read that attribute in two places and
they disagreed: the scene-object AABB reader used `sphere`, the robot-link
reader used `box`. Read as a box, such a geom lost its stated `size` as well as
its shape - the box branch needs three components and a ball declares one - so a
link written `<geom size="0.03"/>` reported a 0.05 m box, and a 60 mm ball
spawned as a 100 mm cube under a successful load. Both readers now resolve the
default through one name, so the two cannot answer one element differently.

A body with no `<geom>` at all is unchanged: there is no geometry to name, so it
keeps the no-geometry box proxy.
