### Fixed: the last nine CodeQL alerts on main

`routes_record.contained_path` uses the positive containment guard (`real == home
or real.startswith(home + os.sep)`) the scanner recognises, so the six
path-injection findings behind the label route and `episode_labels` close; the
record respawn line and the recorder's save failure name the exception type and
leave the exception's words in the log; the collect route logs the dataset root
through `log_safe`; one silent `except OSError` in the training dataset walk
says why it is fine. Behaviour and refusal sentences are unchanged.
