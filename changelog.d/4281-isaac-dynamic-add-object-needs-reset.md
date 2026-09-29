### Fixed: a dynamic Isaac add_object says reset() must come next

Adding a dynamic object to a running Isaac scene invalidates the physics view, so the next `step()` is refused until `reset()`. `add_object` now says so when it happens (text, and `requires_reset` in the json) instead of leaving it to the refusal.
