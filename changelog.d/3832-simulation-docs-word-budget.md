### Docs: the simulation reference is five pages of tables instead of one 10,542-word wall

`docs/simulation/overview.md` had grown to 10,542 words - 66% of it prose about
`run_policy` / `eval_policy` that restates their own docstrings (2,953 and 1,583
words), including the "unvalidated, this returned success" narrative of the pull
request that added each guard. It is now split at its own `##` boundaries into
`overview.md` (the scene, rendering and registry verbs, 1,386 words),
`physics.md` (982), `rollouts.md` (1,928), `observers.md` (635) and
`predicates.md` (609): 5,540 words total, with every parameter, refusal domain
and reported field kept as a table.
