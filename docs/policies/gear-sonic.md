---
description: GEAR-SONIC VLA — third-party provider, integration status.
---

# GEAR-SONIC

GEAR-SONIC is a third-party VLA. There is no built-in `GearSonicPolicy` class on
`main` today; integration is by writing a custom policy that talks to the GEAR-SONIC
API (see [Custom policies](custom-policies.md)).

## Status

> **Not bundled.** No `gear_sonic` provider in `registry/policies.json`. To use
> GEAR-SONIC with `strands-robots`, write a thin `Policy` subclass per the custom
> policies guide.

## Why a separate page?

- The PR40 docs draft mentioned GEAR-SONIC as a planned provider.
- Several issues / discussions reference it.
- This page exists so the link from the navigation tree resolves to a clear "use the
  custom policy mechanism" answer.

## Sketching an integration

```python
# my_gear_sonic.py
from typing import Any
from strands_robots.policies import Policy, register_policy

class GearSonicPolicy(Policy):
    def __init__(self, server_address: str, **kwargs: Any):
        self._server = server_address
        self._keys: list[str] = []
        # initialise your GEAR-SONIC client here

    async def get_actions(self, observation_dict, instruction, **kwargs):
        # POST observation_dict + instruction to the GEAR-SONIC server
        # parse the response into a list of action dicts
        return [{}]

    def set_robot_state_keys(self, robot_state_keys):
        self._keys = robot_state_keys

    @property
    def provider_name(self) -> str:
        return "gear_sonic"

    @property
    def requires_images(self) -> bool:
        return True

register_policy("gear_sonic", lambda: GearSonicPolicy, aliases=["sonic"])
```

After registering:

```python
from strands_robots.policies import create_policy
policy = create_policy("gear_sonic", server_address="http://...")
```

To register permanently in `strands_robots/registry/policies.json`:

```json
{
  "gear_sonic": {
    "module": "my_pkg.gear_sonic",
    "class": "GearSonicPolicy",
    "shorthands": ["sonic"],
    "description": "GEAR-SONIC VLA client."
  }
}
```

## Adopting upstream

If GEAR-SONIC stabilises an inference protocol we'll consider adding a built-in
provider that mirrors how `Gr00tPolicy` works. Track the
[issue tracker](https://github.com/strands-labs/robots/issues).

## See also

- [Custom policies](custom-policies.md) — write the `Policy` subclass.
- [Policy overview](overview.md) — the ABC contract.
- [GR00T](groot.md) — reference implementation for a server-based policy.
- [Cosmos 3](cosmos3.md) — another server-based VLA provider.
