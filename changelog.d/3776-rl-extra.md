### Added: an `[rl]` extra for the from-scratch RL trainers

`strands_robots.training.rl` computes in torch and `GymSimEnv` presents a
`SimEnv` through gymnasium, but neither package was declared by an extra for
that purpose - torch arrived only inside `[kimodo]` / `[lerobot]` and gymnasium
only inside `lerobot` - so on any other install the RL page's first line,
`create_trainer("ppo")`, died on the interpreter's `No module named 'torch'`.
`pip install 'strands-robots[rl]'` now brings torch, gymnasium and the MuJoCo
backend the env adapters step (it is part of `[all]`); the four modules that
imported torch bare bind it through `require_optional(..., extra="rl")`, so an
install without the extra is refused with its name at every door, and the
existing trainer / `GymSimEnv` gates name it too. `docs/training/rl.md` gains
the install line, and the two places the extras pages restate a count outside
their table - the install block's "N-extra bundle" and architecture.md's
pointer at "the N it leaves opt-in" - are derived from the manifest as well,
which corrected both: each had counted `[all]` itself.
