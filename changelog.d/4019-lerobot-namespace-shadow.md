### Fixed: a directory named `lerobot` is no longer reported as a lerobot install

`import lerobot` succeeds for two states that mean opposite things: an install,
and any directory called `lerobot` that Python can see, which imports as an
empty namespace package. Three surfaces read the bare import as the whole
question and reported the second as the first - `doctor` as `PASS lerobot ?`,
the `use_lerobot` discovery path as a catalog of that directory's own files with
`robots (0)`, `teleoperators (0)`, `cameras (0)` and `policies (0)`, and the
draccus registry walk as `lerobot is installed but lerobot.robots is not
importable (partial install?)`. So a host with no lerobot at all read as an
install exposing nothing, which is the answer an agent asking "what does this
LeRobot install expose?" was handed. Reachability now has one owner,
`utils.lerobot_install_error`, which names the directory that resolved and both
remedies; an installed host is unaffected, because a regular package wins over
a namespace portion wherever it sits on the import path.
