---
description: 72 robots across 8 categories. Every name addressable from Robot('name').
---

# Robot catalog

`strands-robots` ships with a registry of **72 robots** across 8 categories. Every robot
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

-   :material-arrow-left-right:{ .lg .middle } **Bimanual** · 4

    ---

    Two-arm rigs.

    [:octicons-arrow-right-24: Bimanual catalog](bimanual.md)

-   :material-human:{ .lg .middle } **Humanoids** · 18

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

-   :material-truck:{ .lg .middle } **Mobile manip** · 5

    ---

    Mobile bases with arms.

    [:octicons-arrow-right-24: Mobile manip catalog](mobile.md)

-   :material-airplane:{ .lg .middle } **Aerial** · 2

    ---

    Quadcopters.

    [:octicons-arrow-right-24: Aerial catalog](mobile.md)

-   :material-emoticon:{ .lg .middle } **Expressive** · 1

    ---

    Social / desktop robots.

    [:octicons-arrow-right-24: Expressive catalog](humanoids.md)

</div>

## Counts at a glance

| Category | Count | Page |
|----------|------:|------|
| Arms | 23 | [arms](arms.md) |
| Bimanual | 4 | [bimanual](bimanual.md) |
| Humanoids | 18 | [humanoids](humanoids.md) |
| Hands | 9 | [hands](hands.md) |
| Mobile | 10 | [mobile](mobile.md) |
| Mobile manip | 5 | [mobile](mobile.md) |
| Aerial | 2 | [mobile](mobile.md) |
| Expressive | 1 | [humanoids](humanoids.md) |
| **Total** | **72** | |


## Add a new robot

Robots are JSON entries in `strands_robots/registry/robots.json`. No code change is
needed for most additions - see [Architecture](../architecture.md)
for the JSON schema and asset-fetch strategies.

## See also

- [Robot factory](../getting-started/robot-factory.md) - the `Robot()` signature.
- [Quickstart](../getting-started/quickstart.md) - pick one,
  spawn it.
