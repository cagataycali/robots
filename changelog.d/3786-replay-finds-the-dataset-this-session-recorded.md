### Fixed: `replay_episode` reads back a dataset this session recorded to a custom `root=`

Naming only the `repo_id` no longer surfaces a raw HuggingFace 404. The replay
reads the directory this sim recorded that id to and names it, a dataset already
at the default location is never shadowed, and a Hub miss that still happens
names the default directory that was tried and the `root=` remedy. The id is
stashed with the root by the one resolve-and-stash every backend goes through,
so a recording is locatable whichever backend made it.
