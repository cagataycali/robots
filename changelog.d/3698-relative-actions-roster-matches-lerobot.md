### Fixed: the relative-action policy roster is the one lerobot declares, offline and in the docs

`extra["relative_actions"]` is gated on whether a policy's lerobot config
declares `use_relative_actions`. Online that is discovered from lerobot's
registry; offline `_RELATIVE_ACTION_POLICY_TYPES_FALLBACK` *is* the gate, and it
had not been widened for `vla_jepa`, whose `VLAJEPAConfig` declares the field.
So with lerobot's registry unavailable the preflight refused a combination the
run itself supports, and named a roster that omitted the type it was refusing:

    validate(policy_type="vla_jepa", relative_actions=True)
    online  -> accepted
    offline -> "relative_actions is not supported by policy_type 'vla_jepa'
                (only ['groot', 'pi0', 'pi05', 'pi0_fast'] expose
                use_relative_actions)"

The snapshot now carries `vla_jepa`, so both paths reach the same verdict and
`act` is still refused by both.

The Training overview's copy of that roster was stale in the other direction and
had been since it was written: it named `pi0` / `pi05` / `pi0_fast` while the
gate had already accepted `groot`, so the page denied a combination `validate()`
allowed. A hand-written roster that no test reads goes stale on the commit that
widens the gate, so the page's rosters - relative actions and quantile
normalization - are now graded against the gates that derive them, and the two
prose copies of the set inside `training/lerobot.py` (a comment one line above
the frozenset, and the `_relative_actions` docstring) point at the live
discovery instead of re-listing it.
