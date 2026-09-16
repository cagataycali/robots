### Fixed: a second `start_recording` no longer drops the live recording's frames

Calling `start_recording` while a recording was live replaced the recorder, and
the frames buffered since its last saved episode were lost without a word (or,
on a schema mismatch, both sessions were lost with `recording` left off). All
three backends that record -- MuJoCo, Isaac and Newton -- now refuse the second
start, naming the live `repo_id`, the frames it has buffered and the episodes
*this session* has saved, plus the remedy (`stop_recording` first, which saves
them). The live recording is untouched.
