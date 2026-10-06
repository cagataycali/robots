### Fixed: `send_action` says why a multi-element value was refused

A mapping value that is not a scalar was refused with "must be a scalar number
..., got list.", which reads as "lists are not allowed" even though a
one-element `[v]`, `(v,)` or `np.array([v])` is unwrapped to its scalar and
accepted. The refusal now names how many elements the value carries and that a
one-element sequence is unwrapped once, so `[0.5, 0.6]` points the caller at
one value per actuator key. Mappings and strings keep the short message.
