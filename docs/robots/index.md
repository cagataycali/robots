---
description: Every robot in the registry, by category. Every name addressable from Robot('name').
---

# Robot catalog

`strands-robots` ships with a registry of **{{n:robots}} robots** across {{n:categories}} categories. Every robot
is addressable by name through the factory:

```python
from strands_robots import Robot
sim = Robot("panda")
sim = Robot("unitree_g1")
sim = Robot("aloha")
```

## Browse by category

<div class="grid cards" markdown>

-   :material-arm-flex:{ .lg .middle } **Arms** · 23

    ---

    Single-arm manipulators.

    [:octicons-arrow-right-24: Arms catalog](arms.md)

-   :material-arrow-left-right:{ .lg .middle } **Bimanual** · 5

    ---

    Two-arm rigs.

    [:octicons-arrow-right-24: Bimanual catalog](bimanual.md)

-   :material-human:{ .lg .middle } **Humanoids** · 19

    ---

    Full-body humanoids.

    [:octicons-arrow-right-24: Humanoids catalog](humanoids.md)

-   :material-hand-back-right:{ .lg .middle } **Hands** · 9

    ---

    Dexterous end-effectors.

    [:octicons-arrow-right-24: Hands catalog](hands.md)

-   :material-car-sports:{ .lg .middle } **Mobile** · 10

    ---

    Quadrupeds + wheeled bases.

    [:octicons-arrow-right-24: Mobile catalog](mobile.md)

-   :material-truck:{ .lg .middle } **Mobile manip** · 6

    ---

    Mobile bases with arms.

    [:octicons-arrow-right-24: Mobile manip catalog](mobile-manip.md)

-   :material-airplane:{ .lg .middle } **Aerial** · 2

    ---

    Quadcopters.

    [:octicons-arrow-right-24: Aerial catalog](aerial.md)

-   :material-emoticon:{ .lg .middle } **Expressive** · 1

    ---

    Social / desktop robots.

    [:octicons-arrow-right-24: Expressive catalog](humanoids.md)

</div>

## Drivable for real

Generated from `strands_robots/registry/robots.json` and the native-driver table in
`strands_robots/drivers` by `docs/hooks/coverage_matrix.py` at build - do not edit. One row
per registered robot: the lerobot robot type the registry declares, the native driver this
package registers, both, or neither - and neither is simulation-only until one of the two
arrives. Which of the two `driver="auto"` picks is
[the factory's answer](../getting-started/robot-factory.md#choosing-a-driver); what a native
driver must satisfy is on [Native drivers](../hardware/native-drivers.md).

{{coverage_matrix}}

## Add a new robot

Robots are JSON entries in `strands_robots/registry/robots.json`. No code change is
needed for most additions - see [Architecture](../architecture.md)
for the JSON schema and asset-fetch strategies.

## See also

- [Robot factory](../getting-started/robot-factory.md) - the `Robot()` signature.
- [Quickstart](../getting-started/quickstart.md) - pick one,
  spawn it.
