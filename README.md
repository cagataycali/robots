# start_policy `policy_kwargs` refusal

`capture.py` drives the real MuJoCo engine three times (at rest / an accepted
`start_policy` rollout / a `start_policy` carrying an unsplattable
`policy_kwargs`), renders each and writes `facts.json`. `compose.py` builds the
figure, re-deriving every cell from that dump and asserting each claim -- the
honored panel differs from at-rest on >10% of pixels, and the refused panel is
pixel-identical to at-rest -- before saving.

`mutation_table.py` mutates the guard two ways (delete it; keep the call and
discard the `return err`), runs the new case and the module's pre-existing cases
against each, and restores the source byte-identically.
