### Fixed: a nested pytest over a `tmp_path` names that directory as its rootdir

Two test files spawn a child `pytest` over a file written into `tmp_path` and
named no rootdir, so pytest resolved one by its default rule - the common
ancestor of the child's cwd and its arguments. For
`tests/test_session_truncation_is_reported.py` that was the xdist worker's base
temp, the directory every `tmp_path` that worker has made so far lives in, and
the session's root `Dir` collector lists the whole of it to reach the one target
directory underneath: 0.18 ms per sibling entry on a hosted runner, which put
the file's seven cells at 8-12 s each late in a worker's run and made it the
largest file in the suite (#3869). `tests/test_abandoned_work_item_cannot_hang_exit.py`
had the same shape once, rooted at the ancestor of the repository and the base
temp. Both children now pass `--rootdir` naming the target; with 24,000 planted
siblings the seven cells go from 19.7 s to 4.7 s (pin included), and with none
they are unchanged. Pinned by the node-id shape a `--collect-only` run prints,
which is relative to the rootdir, and by the rootdir the child's session header
names.
