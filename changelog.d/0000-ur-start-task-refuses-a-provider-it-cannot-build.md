### Fixed: a policy provider `URDriver.start_task` cannot build is refused, not raised

`URDriver.start_task` is the fleet's only `start_task` that builds a policy from
the provider registry -- the other twelve drivers refuse the verb outright -- so
it is the only one where a build failure reaches a caller. It is invoked as an
agent tool, where every verb promises a status envelope, and its docstring
already documented the answer: "a refusal naming the provider that could not be
built". The handler named `(ImportError, TypeError, ValueError)`, which covered
neither half of the real population. Driven over every spelling the policy
registries hold, 9 of 29 raised past the envelope instead of refusing:
`lerobot_local` and its `lerobot` alias among them, because
`create_policy`'s one documented exception, `UntrustedRemoteCodeError`, is a
`RuntimeError` -- and it fires under the *secure* default, with
`STRANDS_TRUST_REMOTE_CODE` unset.

Widening the tuple would not have settled it. A provider whose constructor
resolves a checkpoint off disk raises `FileNotFoundError` from a path the caller
mistyped (`policy_provider="rl"`, `checkpoint_dir="/nonexistent"`), an `OSError`
outside any tuple sized for today's classes, and `register_policy` lets a caller
add a provider this package never sees. The build is now a total recovery path,
the way the rollout loop in the same file already treats a policy that raises one
step later. All 29 registered spellings now answer with an envelope; the 9 that
build and roll out are unchanged.
