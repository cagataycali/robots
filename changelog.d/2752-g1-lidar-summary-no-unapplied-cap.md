### Fixed: a G1 LiDAR summary no longer reports a point cap it never applies

`G1Driver._on_lidar_cloud` builds the record the mesh publishes on
`strands/<peer>/lidar/summary`. Every field in it is read from the
`PointCloud2_` header - width, height, point_step, row_step - so no point is
ever enumerated and nothing is downsampled. The record nonetheless carried a
`capped_at` field copied from a `lidar_max_points` constructor parameter, and
that parameter had exactly one reader: the line that copied it into the field.
A Mid-360 frame therefore published `count: 24000` beside `capped_at: 4000`,
telling a consumer that the number next to it had been capped at 4000 when it
was the cloud's true uncapped size. Setting the knob changed only that claim.
The comment defending the constant named `_summarise_cloud`, a method with no
definition anywhere in the tree.

The field and the parameter are gone; a caller still passing `lidar_max_points=`
is accepted and logged at debug, because the driver already ignores extras so
the factory can forward them. `count` is deliberately unchanged: it is the
cloud's true size, and a Mid-360 that drops from 24000 points to 3000 is
reporting a fault that clamping the number would hide. What bounds the record is
its fixed, header-derived shape - the same for a 200-point frame and a
30k-point one - which is now what the docstring says and what the tests assert.
