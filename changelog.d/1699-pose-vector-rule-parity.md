### Fixed: every entry point that takes a pose applies the same pose-vector rule

`move_to` takes the same world-frame `[x, y, z]` and the same wxyz quaternion as
`add_object` / `move_object` / `add_camera` / `add_robot`, but hand-rolled its
own check instead of using the shared pose-vector guard, and the two contracts
had drifted apart in both directions.

`move_to` computed `len(position)` directly, so a value with no length raised a
bare `TypeError` straight through the `{"status": "error"}` contract its own
docstring promises it never breaks:

```python
sim.move_to(robot_name="panda", position=np.float64(0.4))
# TypeError: object of type 'numpy.float64' has no len()
```

A scalar, a NumPy 0-d scalar and a generator (from `map` or a comprehension over
an observation) all landed there, for `position` and for `orientation`. The
scene-construction calls refuse every one of them cleanly.

In the other direction, the shared guard accepted a `bool` component, because
`float(True)` is `1.0`. `add_object(position=[True, 0, 0.3])` reported success
and compiled the body a metre out on x; `add_camera`, `add_robot` and
`move_object` took it too. `move_to` refused it, and so does the agent-tool
router - the direct API was the only surface that accepted it.

The guard is now `strands_robots.utils.coerce_pose_vector` (with
`pose_vector_error` / `finite_vector_error`), promoted out of the MuJoCo facade
because the motion primitives live in a module that facade imports, which is
what made sharing it impossible. It refuses a `bool` where a coordinate, extent
or colour channel belongs, matching the router, and `move_to` routes both of its
pose parameters through it. A NumPy array pose is still accepted everywhere, and
a supplied quaternion still constrains the IK solve.
