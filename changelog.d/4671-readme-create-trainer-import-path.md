### Fixed: the README names `create_trainer` by the path it imports from

The README's Train row wrote `create_trainer("lerobot_local")` as a bare name,
so a reader who tried `from strands_robots import create_trainer` got an
`ImportError`. The row now writes
`strands_robots.training.create_trainer("lerobot_local")`, and a test checks
that every call the README writes in inline code resolves: a bare name from
`strands_robots`, a dotted one from its module.
