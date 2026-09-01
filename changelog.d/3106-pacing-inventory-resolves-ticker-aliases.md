### Fixed: the paced-loop inventory can see a ticker imported under an alias

`tests/test_mesh_pacing_ticker.py` requires every paced loop to hand its
ticker's release to the language, because the ticker owns a selector and a
socketpair and a release each loop has to remember is one a loop eventually
forgets. It found its population by matching the local name `Ticker`, so a
ticker imported under any other name was not merely ungraded - it was
invisible, and the sweep reported a clean tree while a bare construction sat in
it.

That is not a hypothetical: a module binding `Ticker` at module scope for an
annotation has to import it under another name to construct one at runtime, so
the alias is a side effect of correct code and nothing at the call site looks
wrong. `PolicyRunner.run` is the live case - measured over the package, the
scan saw 14 constructions and all 14 were acquired, while a scan resolving
aliases finds a 15th that was not.

The inventory now resolves import aliases, and accepts
`stack.enter_context(Ticker(...))` alongside `with Ticker(...) as ticker:` as
structural acquisition. The second half is what the first half needs: the
rollout runner is the package's first *conditional* pacer - it paces only when
real-time pacing was asked for, and its ticker is read inside a closure
spanning both step loops - so it has no single block to bind the ticker to, and
`ExitStack` is the standard library's form for exactly that. Both spellings put
the release in the interpreter, which is the whole content of the rule.

`PolicyRunner.run` acquires its ticker on an `ExitStack` accordingly. Its
release was already correct on every path, including the error-dict return, so
this closes a coverage hole rather than a leak: the exposure was that a future
edit to a 431-line `try` could drop the `close()` and the guard that exists to
catch that would have stayed green.
