### Fixed: a refused resize no longer keeps the colour and friction from its own call

`set_geom_properties` takes `color`, `friction` and `size` together - the shape
`docs/simulation/domain-randomization.md` shows for perturbing one manipuland - and
records all three in the scene spec before it writes the compiled model, so the
values it reports survive the next recompile. A resize is then refused on evidence
that only exists once the new size is recorded: the owning body's inertial row is
re-derived from a compile of the persisted spec, and a shrink whose body would weigh
less than MuJoCo's `mjMINVAL` does not compile at all. That refusal rolled back only
the `size`, so the colour and friction from the same call stayed applied, in the
model and in the spec the next recompile restores the geom from.

The result the caller reads is `status="error"` naming the resize, so nothing points
at the two properties that were kept. Measured on a density-derived body, one call
carrying `color`, `friction` and a size too small to compile: the refusal message is
byte-identical with and without this change, and before it the geom was left with the
requested `geom_rgba` and `geom_friction` and still carried both after an unrelated
`add_object` recompiled the scene. The docstring already promised the opposite - an
error result "leaves the model untouched, rather than ... applying a
shape/appearance the caller never asked for" - and the rollback's own comment claims
it leaves the two representations "in step".

No model write now happens until every part of the call has passed, so a refusal
leaves the model untouched however many properties one call carries rather than only
the resize it names, and the spec rollback restores all of them. That rollback can
itself fail on a spec that no longer agrees with the compiled model, which is exactly
the divergence this path exists to prevent; its reason was discarded and is now
reported alongside the refusal. A resize the scene can compile still applies all three
durably, and a size-only refusal is unchanged.
