### Changed: the Learn pages say what the code does

`hardware/drivers.md` names one default for `driver=` (`auto`, falling back to
lerobot) and says what 0.8 removes (`start_task` building its own policy, not the
method); `hardware/feetech-arms.md` says the lerobot driver takes degrees;
`simulation/index.md` says when to use `Robot()` and when `create_simulation()`
and that the engine's `mesh` keyword takes a handle, off unless given (18 fences
drop a redundant `mesh=False`); `security.md` links the operator gate instead of
restating it and names the ungated native `move_to`; 20 pages gain
`description:` front matter, paid for by cuts on the same pages (#4289).
