### Fixed: the docs backend tables list `mjlab`, and Isaac is no longer called a plugin

`docs/concepts/backends.md` and `docs/learn/simulation/index.md` listed three
simulation backends while `create_simulation` ships four; both tables now carry
the `mjlab` row, and the concepts page no longer sends Isaac users to a
`strands-robots-sim` plugin (the built-in `isaac` backend installs through
`strands-robots[sim-isaac]`). A test holds both tables to
`_BUILTIN_BACKENDS`.
