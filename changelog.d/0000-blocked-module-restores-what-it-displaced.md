### Fixed: blocking an optional dependency in a test restores what it displaced

Making a dependency missing for one block takes two entries -- `sys.modules`,
and the `require_optional` memo -- and three files had copied the idiom by hand.
Two of the copies ended their `finally` with `del sys.modules["imageio"]`, which
removes the key rather than putting back the module the interpreter had, so the
next import returned a *different* object: a sibling's `monkeypatch.setattr`
then landed on a module the code under test never reached.
`tests/simulation/test_policy_runner_video_writer_cleanup.py` failed that way
against a runner that does close its writer, while passing in isolation.

`tests._blocked_module.blocked` is now the one owner of the idiom and restores
both mappings exactly, whether the block returns or raises. The tree-wide rule
in `tests/test_sys_modules_removal_leaves_no_orphan.py` no longer reads the mere
presence of a `finally` as restoration: it asks, per key, whether the displaced
value was captured, and it reads a `setup_method`/`teardown_method` pair as one
unit. On the tree as it stood, that rule reports the two offenders where the old
one reported none.
