### Fixed: a real-robot task reports how long it ran, however it ended

The elapsed time is now settled wherever a task's terminal state is recorded,
so the reply and every `status` afterwards carry the real figure. Before, only
a rollout that ran to its own budget reported one: a task stopped from outside,
stopped during bring-up, ended by `cleanup()`, or failed on a connect, a policy
that would not initialize or a raise mid-rollout all reported `0.0s` beside a
non-zero step count -- and kept reporting it, since nothing writes the duration
once the task is terminal.
