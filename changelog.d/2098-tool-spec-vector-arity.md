### Fixed: the MuJoCo tool_spec publishes the vector component counts the router enforces

`_validate_and_build_kwargs` refuses any vector parameter whose component count
is not in `_VECTOR_PARAM_LENGTHS` - `Parameter 'orientation' must be a list of 4
numbers, got 3.` Ten `tool_spec.json` properties are validated that way and all
ten were advertised as an unbounded `array` of `number`, so the one figure a
model needed to form a valid call was the one figure the schema never published,
and the arity was discoverable only by being rejected.

`minItems` / `maxItems` are now declared from the router's own table: 3 for
`position`, `target`, `origin`, `force`, `torque_vec`, `gravity`, `direction`
and `point`, 4 for `orientation`, and 3..4 for `color`, which accepts rgb or
rgba. No accepted count changed.

`orientation` also gains a description, because a count cannot pin it:
`[w, x, y, z]` and `[x, y, z, w]` are both four components, `add_object` assigns
the value straight to `body.quat`, and MuJoCo reads that scalar-first - so the
wrong order passes every check the router has and applies a different rotation
under `status="success"`. The convention was already documented for human
readers in `docs/simulation/overview.md` and commented in the router's table;
the agent-facing schema was the one place that omitted it.
