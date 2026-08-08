### Added: the policy registry is pinned to agree with the provider constructors

`policies.json` describes each provider with a `config_keys` list and an optional
`defaults` map, and `build_policy_kwargs` merges both into one dict that
`create_policy` splats into the provider class. Nothing compared either against a
constructor signature, so a key that stopped being a parameter stayed advertised.

The two sources fail differently, and `defaults` is the worse of them.
`config_keys` is a filter, so a stale entry only bites when a caller passes that
key. The defaults loop applies no such test:

```python
for key, default_val in defaults.items():
    if key not in kwargs:            # no `key in allowed_keys` here
        kwargs[key] = default_val
```

So an orphaned `defaults` key is forwarded on *every* call with no caller
involvement - for `cosmos3` and `vera`, which declare no `**kwargs`, that is a
`TypeError: __init__() got an unexpected keyword argument` for every
`create_policy(provider)` rather than for one unlucky caller. The remaining ten
providers swallow it into a `**kwargs` nothing reads, which is the inert-knob
shape rather than a safer outcome.

A third disagreement is silent in both shapes: a `defaults` key absent from
`config_keys` is still applied, while a caller's own value for that key is
dropped by the filter - an override that does nothing, with no error.

All three are now pinned across every provider, in the strict form (a key must
name an explicit parameter), which is what all twelve already satisfy. The keys
are derived from the registry rather than listed, so a provider that gains one is
covered as soon as it is declared, and a non-vacuity test names any provider
whose signature could not be read instead of counting it as consistent.

No behaviour change: the registry and the constructors agree today, in all three
directions, for all twelve providers. This closes the general form that #2013 left
open after pinning the property for `vera` alone.
