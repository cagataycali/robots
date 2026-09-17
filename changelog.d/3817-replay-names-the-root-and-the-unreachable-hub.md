### Fixed: a dataset `replay_episode` cannot open names the directory it read

Two ways a dataset is simply not there still surfaced as a raw library error.
A `root=` the caller passed themselves was reported as
`404 Client Error ... Request ID ... make sure you specified the correct
repo_id and repo_type` — naming the one input they chose nowhere in the reply —
and a Hugging Face Hub that could not be reached read
`[Errno 111] Connection refused`, naming neither the dataset, the directory,
nor the Hub. Both now name the directory that was read, the verdict, and the
datasets that ARE on disk there (the answer to a typo, or to a `root=` aimed
one level too high). An unreachable Hub keeps the library's own text in
parentheses — it names the endpoint, and offline mode names the variable to
unset. Episode-out-of-range and every other load error stay verbatim.
