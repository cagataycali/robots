### Quality: the test suite executes each docs hook once per worker

Forty test modules loaded a `docs/hooks` file by path themselves, several on
every call, so each fresh module threw away the hook's cached scan and paid it
again - `env_vars` parses the whole package, about 12 s a render. They now share
one loader, `tests._docs_hooks.docs_hook`, and a grader refuses a test module
that executes a hook file on its own. The 40 files ran in 164.5 s on two workers
before and 63.8 s after.
