### Fixed: a misspelled `composite` or `persistent` provider is offered the right spelling

`create_policy` builds its "Did you mean" hint from `list_providers()` and
`list_aliases()`, and neither reports the two auto-discovered wrappers. So
`create_policy("composie")` suggested `'cosmos3'` and `create_policy("persistant")`
suggested nothing. The search now includes both wrappers: `composie` is offered
`'composite', 'cosmos3'` and `persistant` is offered `'persistent'`. The two
listing functions report what they reported before.
