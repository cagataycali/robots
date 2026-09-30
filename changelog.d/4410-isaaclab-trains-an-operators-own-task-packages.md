### Added: the `isaaclab` trainer trains and plays tasks from the operator's own Isaac Lab task packages

Isaac Lab trains a task from an outside package when `--external_callback
<module>.<function>` registers it, but strands could not reach one: `validate()`
scanned only `isaaclab_tasks` ("... is not registered ... did you mean
['Isaac-Cartpole-Direct']?"), `extra` had no key for the callback, and the Isaac Lab
child drops `PYTHONPATH` on purpose. The operator now names their packages in
`$STRANDS_ISAACLAB_TASK_PACKAGES` (`module:register_fn`, comma-separated, importable
in the Isaac Lab venv - installed or on a `.pth` there; malformed entries are
skipped): their `gym.register` ids join the task check, and a task they register
gets `--external_callback module.register_fn` on `train` and on `play`. A package
named without its function is refused with the spelling to use. The variable is
operator-owned like `ISAACLAB_PYTHON`: an agent can train those tasks but cannot
point the Isaac Lab process at other code.
