### Fixed: `send_action` says how many actuators it commanded and how far the world moved

The success text was `Action applied to '<robot>' (N keys).`, so an empty
mapping read as a no-op even though the world still advanced `n_substeps`.
MuJoCo, Newton and Isaac now return the same sentence from one helper,
`Commanded N actuator(s) on '<robot>'; advanced K physics substep(s).`, and an
empty mapping adds a pointer to `step(n_steps=...)`. Status and behaviour are
unchanged.
