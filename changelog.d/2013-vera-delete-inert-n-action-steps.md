### Removed: the inert VERA `n_action_steps` knob

`VeraConfig.n_action_steps` was documented twice as the "deploy chunk size
(actions executed per infer)" and read by nothing. It was public in four
spellings at once - a `VeraPolicy` keyword, a `VeraConfig` field, a
`VERA_N_ACTION_STEPS` deploy variable, and an entry in the `vera` provider's
`config_keys` - and each of them reported success. `VeraPolicy(n_action_steps=8)`
constructed, stored `8`, and executed chunks of whatever length the server
returned; `-7` and `"eight"` were stored just as happily, and the launched server
argv was byte-identical in every case, in both the subprocess and docker launch
modes.

The chunk length is a server-side quantity: `_infer` returns the raw `[H, D]`
array the server sent and `_chunk_to_action_dicts` maps all `H` rows into the
queue, so there was no local slicing step for the field to be the width of. The
neighbouring `sample_steps` shows what wired-up looks like on the same dataclass
- it is forwarded as `--sample-steps` and as `VERA_SAMPLE_STEPS` - and
`n_action_steps` appeared in the server runner not at all.

It is deleted rather than validated. A value domain would have refused `-7` and
then still honored nothing, which is a worse contract than an unvalidated knob,
not a better one; that is why #2012 settled the neighbouring `render_width` on
the shared media pixel domain and deliberately left this field alone.

**Breaking:** passing `n_action_steps` to `VeraPolicy` or `VeraConfig`, or
through `create_policy("vera", ...)`, now raises `TypeError` instead of being
accepted and ignored. Every such caller was already getting no effect from it,
and the deploy variable is now simply unread. Removing the registry entry in the
same change is load-bearing: `VeraPolicy` takes no `**kwargs`, so a `config_keys`
entry left behind would have turned a silently-ignored value into a `TypeError`
on the factory path. The general form of that registry-vs-signature guard is
tracked as #2022.
