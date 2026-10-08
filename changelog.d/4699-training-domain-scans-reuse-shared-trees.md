### Tests: the training domain graders parse each trainer module once per process

The shared-domain table and six per-field domain graders parsed every training
module afresh on each question they asked of it - 179 cells, each re-parsing
25 modules several times. They now ask the process-wide shared tree
(`tests._package_ast.parse_source` / `walk_tree`) instead, with the same
verdicts and planted negatives; `test_every_shared_domain_has_one_owner.py`
runs in 6.5 s instead of 22.9 s, 4.1 s of which is the shared tree build any
earlier grader in the worker has already paid.
