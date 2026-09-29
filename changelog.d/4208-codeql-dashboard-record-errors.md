### Fixed: a recorder failure reaches the dashboard as a reason or a kind, never as exception text

`DatasetRecorder.save_episode` and `push_to_hub` still answer an error with
`message` (their full words, exception included, which the run_policy tool and
scripts read) and now also with `reason` (the recorder's own sentence for a
refusal it decided itself) or `error_type` (the class of the exception a call
raised). The dashboard's record worker builds the operator's sentence from
those two and logs `message`, so a Hub URL with a token in it or a local path
in a traceback stays in the log. `contained_path` keeps the same acceptance in
the two-test shape a path checker reads as a barrier, `collect_episodes` logs
its `dataset_root` through the one-line redactor, and the last bare `pass` in
the dataset scan gets its comment. Closes the 9 CodeQL alerts the scan of
`main` left or opened after #4136.
