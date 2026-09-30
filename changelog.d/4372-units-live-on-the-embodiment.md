### Fixed: `lerobot_local` refuses `state_units` / `action_units` as constructor keywords and names where they go

The two names are embodiment fields; passed to the constructor they were dropped with a warning and the rollout reported success with the unit frame unchanged. They are now refused before the trust gate with the embodiment shape and the two frames (`degrees`, `native`); `radians` is not a frame because `native` already means what the robot emits. The lerobot_local page says the same. (#4164)
