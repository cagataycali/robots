### Fixed: a navigation goal the bridge cannot honor is refused instead of sent

`RosBridgedRobot.navigate_to` hands a goal pose to the robot's own navigation
stack, so the coordinates are the whole command - and none of them was
validated. `use_ros` cannot cover this: it checks the action name and interface
type and (since the transport-option guards) the `timeout`, but the pose travels
inside the request body, which it forwards verbatim.

```python
robot.navigate_to(x=float("nan"), y=0.0)   # status 'success'
# -> goal sent: position {'x': nan, 'y': 0.0}
robot.navigate_to(x=float("inf"), y=0.0)   # status 'success', inf on the wire
robot.navigate_to(x=1.0, y=2.0, yaw=float("nan"))
# -> goal sent with orientation {'z': nan, 'w': nan}: not a rotation at all
robot.navigate_to(x=1.0, y=2.0, yaw=float("inf"))
# ValueError: math domain error   (raised past the result dict)
robot.navigate_to(x=None, y=2.0)
# TypeError: float() argument must be a string or a real number, not 'NoneType'
robot.navigate_to(x=True, y=2.0)           # status 'success', goal at x = 1.0 m
```

A non-finite coordinate serializes as a valid IEEE-754 float64, so the transport
accepts the goal and the planner receives a target it cannot resolve, with
nothing reported at the call site. A non-finite heading is worse than
unresolvable: the planar-quaternion encoding either produces `{z: nan, w: nan}`,
which no controller can normalize into a heading, or raises out of `math.sin`
for an infinite angle - and `navigate_to` is bound as a `navigate_*` agent tool,
where raising escapes the `{"status": ...}` dispatch contract entirely.

`navigate_to` now returns an error result naming the parameter - sending no goal
- for an `x`, `y` or `yaw` that is not a finite number. The rule is the shared
`strands_robots.utils.finite_number_error` already used for `drive`'s velocity
components, since a goal coordinate and a velocity are both signed physical
quantities: both signs stay valid, and a regression test pins that the two
methods return the same verdict for the same value so the domains cannot drift.
The existing "no `nav_action` configured" report still precedes the pose check,
and every goal that worked before is unchanged.
