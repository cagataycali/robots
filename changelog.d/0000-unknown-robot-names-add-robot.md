### Fixed: a registered robot that is not in the scene is named as such, with the add_robot call

`Robot("so101")` then `run_policy("so100")` used to answer `Did you mean: so101?`,
as if `so100` were a typo, though it is a registry robot of its own; `g1` beside a
loaded `so101` got no hint at all. Every "robot not found" error now says
`'so100' is a registered robot but is not loaded in this scene; add it first with
action='add_robot', name='so100'.`, and names the canonical registry robot when
the caller used an alias (`g1` -> `unitree_g1`). Typos and a robot loaded under an
alias keep their "Did you mean".
