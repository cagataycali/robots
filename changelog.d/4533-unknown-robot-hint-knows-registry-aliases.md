### Fixed: an unknown robot name that is a registry alias of a robot in the world is suggested

`robot_action_keys("g1")` on a world holding `unitree_g1` said only "Robot 'g1'
not found." The close-match hint compared spellings, and `g1` is too far from
`unitree_g1` to pass the cutoff even though the registry lists it as an alias.
The hint now suggests every robot in the world that resolves to the same
registry robot first (`g1`, `g1_wbc`, and the reverse case), then the spelling
matches as before. The MuJoCo engine's copy of this message is gone; it uses
the shared `SimEngine` one.
