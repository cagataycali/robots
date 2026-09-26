### Docs: one page documents the remote-inference client, its read deadlines included

`docs/policies/remote.md` restated `docs/inference/remote.md` in 316 words, and
the nav carried both. It was also the only page that named `connect_timeout` and
`request_timeout`, two of the five `config_keys` the `remote` provider accepts,
so the page the provider matrix, the nav and the `strands_robots.inference`
package docstring all point at named three of the five keys and neither read
deadline. The stub is gone and the two deadlines, their defaults (`10.0` and
`60.0` seconds) and their positive-finite-number refusal now sit beside the
`endpoint` / `host` / `port` domains they belong with, paid for by five
derivations deleted from that page - the kernel's part in refusing a client
`port=0`, the lazy-resolution restatement, the reason advertised metadata shares
the local domain, why nothing is coerced, and the two premises behind the
forwarded `reset(seed)`. The page ends one word shorter than it started, the
site loses 317 words and a page, and the site ceiling in
`tests/test_docs_pages_are_within_the_word_budget.py` drops to 113,264 so the
room is banked rather than spendable. New
`tests/inference/test_remote_config_keys_are_documented.py` grades the page
against the registry entry and `RemotePolicy.__init__`, so a knob or a default
that drifts from the page reds; `tests/test_docs_policy_nav_coverage.py` now
accepts a provider page that lives beside its subject instead of requiring a
second page under `docs/policies/`. Towards #3818.
