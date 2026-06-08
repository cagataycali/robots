---
description: GEAR-SONIC VLA — third-party provider, integration status.
---

# GEAR-SONIC

> **Not bundled.** No `gear_sonic` provider in `registry/policies.json`. Write a thin `Policy` subclass per the [custom policies guide](custom-policies.md).

```python
# my_gear_sonic.py
from strands_robots.policies import Policy, register_policy

class GearSonicPolicy(Policy):
    def __init__(self, server_address: str, **kwargs):
        self._server = server_address
        self._keys: list[str] = []

    async def get_actions(self, observation_dict, instruction, **kwargs):
        # call GEAR-SONIC server here, parse response into list[dict]
        return [{}]

    def set_robot_state_keys(self, keys): self._keys = keys

    @property
    def provider_name(self) -> str: return "gear_sonic"

    @property
    def requires_images(self) -> bool: return True

register_policy("gear_sonic", lambda: GearSonicPolicy, aliases=["sonic"])
```

```python
from strands_robots.policies import create_policy
policy = create_policy("gear_sonic", server_address="http://...")
```

## See also

- [Custom policies](custom-policies.md)
- [Policy overview](overview.md)
- [GR00T](groot.md) — reference server-based policy implementation.
