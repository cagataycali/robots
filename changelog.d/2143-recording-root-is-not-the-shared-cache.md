### Quality: a recording unit test's dataset root is its own, not the developer's

`DatasetRecorder.create` and every backend's `start_recording` resolve a
`repo_id` with no `root` to `$HF_LEROBOT_HOME/{repo_id}` - by default
`~/.cache/huggingface/lerobot/{repo_id}`. 39 call sites across nine test
modules relied on that shorthand. They inject a fake dataset class and write
nothing to the shared cache, but `_prepare_create_target` resolves and
*inspects* that path before the fake is reached, so a unit test's verdict
depended on what the developer's cache already held.

Instrumenting `_lerobot_home` across the unit suite recorded 65 test instances
in four modules resolving the shared home, 58 of them onto the single id
`local/probe` - a name any scratch script reaches for. One unrelated dataset
planted there took the fps-domain and frame-shape suites from 133 passed to
22 failed / 111 passed, every failure the same `FileExistsError` naming a
directory in `$HOME` rather than the test's own resolution, which is what makes
it hard to attribute.

Six of the offenders passed `repo_id` *positionally*
(`DatasetRecorder.create("user/data", joint_names=["j1"])`), which is invisible
to a keyword-only rule; one dataset planted at `user/data` took
`tests/test_dataset_recorder.py` from 4 failed to 10 failed.

Every site now passes `root=str(tmp_path / "dataset")`, including the nine that
are refused before the root is resolved - requiring it of those too keeps the
rule one line with no exemptions, instead of modelling which guard fires first.
`tests/test_recording_root_is_not_the_shared_cache.py` reads both the keyword
and the positional form and resolves each module's `_create(**kwargs)` funnel by
AST rather than by name. The rule is keyed on the call site rather than on the
resolution because rebinding the dataset home suite-wide would also break the
one test that legitimately asserts the real default. Afterwards exactly one test
resolves the shared home - that fallback test, which compares a path and reads
nothing - and with a stray dataset planted at all six ids the suite's failure
set is unchanged.
