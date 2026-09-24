### Fixed: two concurrent asset fetches no longer corrupt one description cache

Importing `robot_descriptions.<name>_mj_description` clones an upstream repository
into a shared cache directory, and 40-odd descriptions name one
`mujoco_menagerie`. Upstream's `clone_to_directory` tests that directory for a
usable clone and then creates it, holding no lock, so two callers inside that
window both clone into one directory and the loser's `git` fails in a tree the
winner is building -- reported as the robot's own failure rather than as the other
caller's clone. Measured with two processes on a cold cache: `download_robots`
logged `robot_descriptions failed for panda: ... Unable to create
'.../.git/shallow.lock': File exists` and fell back to a second full clone (40.1s
against 24.1s), and `discover_robot` let a `GitCommandError` escape a registry
lookup that documents `None`, because it catches `ImportError` alone.

The four places the package triggers a clone now import descriptions through
`strands_robots._description_cache.import_description`, which holds an exclusive
lock on the cache directory it is about to clone into: the first caller clones
while the rest wait, and each of them then finds the finished clone. Both
processes now report `downloaded`, both in 24.0s. The guard is best-effort by
design, because a clone is worth more than the serialization of it -- where
`fcntl` is absent or the cache directory cannot hold a lock file, the import
proceeds unguarded.
