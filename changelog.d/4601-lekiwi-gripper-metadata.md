### Fixed: LeKiwi's gripper is resolved from the registry, not guessed from its name

The `lekiwi` entry labelled its `Jaw` joint `gripper` but declared no `gripper`
block, so `set_gripper` found the jaw by the name heuristic and assumed the
default close=low convention. The entry now names the `Jaw` actuator and its
closed/open ends like `so100`, so a renamed actuator in the third-party asset
is a clear error instead of a silent guess.
