### Docs: three `STRANDS_TRAIN_*` variables the package reads are now in the README, and the grader that missed them follows a module constant

`tests/test_env_vars_the_package_reads_are_documented.py` holds every
`STRANDS_*` variable the package reads to a row in the README's Environment
variables table, and recognised a key only as a string literal at the read
site. A module that binds the name once (`RDZV_TIMEOUT_ENV =
"STRANDS_TRAIN_RDZV_TIMEOUT_S"`) and reads through the binding was invisible
to it, as was a read off a conditional receiver (`(env if env is not None
else os.environ).get(...)`, the injectable-mapping idiom). Three variables
reached the environment only that way and appeared in no page:
`STRANDS_TRAIN_EXTRA_FLAGS_ALLOW`, the allowlist a headless `lerobot_train`
refusal names as its own remedy; and `STRANDS_TRAIN_RDZV_TIMEOUT_S` /
`STRANDS_TRAIN_LOCAL_ADDR`, the rendezvous bound and `MASTER_ADDR` override
on a multi-GPU training launch. The walk now follows a `Name` key to a
module-scope string constant and treats a conditional receiver with an
`environ` arm as the environment - with the test change alone it failed on
`main` naming exactly those three - and the README gains a row for each.
