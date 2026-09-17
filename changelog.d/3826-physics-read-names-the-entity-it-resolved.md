### Fixed: a physics read says which entity a bare name resolved to

`_resolve_mj_name` retries a caller's bare name under every robot namespace and
returns the first match. The physics *writes* now refuse such a name when several
robots carry it; the three readers deliberately keep the retry, on the stated
grounds that a read "names one entity and the caller can ask again". The answer
named the *request*, not the entity: in a scene holding `alice` and `bob`,
`get_body_state("base")` reported `Body 'base' (id=1)` with `alice/base`'s pose,
`get_jacobian(body_name="base")` reported `Jacobian for body 'base'` beside a
`dof_joint_names` list spanning both arms, and `forward_kinematics("base")`
reported `FK for 'base'` - so nothing said a choice had been made, which robot
won it, or that a second spelling existed to ask about. A caller building against
a bimanual scene reads one arm forever and cannot tell.

Each read now appends the name it resolved to (`resolved 'base' to 'alice/base'`),
read back off the id the answer describes so the two cannot disagree, and - when
several robots carry the name - who else carries it with the qualified spelling
that reads them (`qualify the name to read another ('bob/base')`). Unchanged:
every read still succeeds, every returned `json` payload is byte-identical, a
qualified name and a name the scene carries verbatim get no note (the retry did
not decide those), and the writes still refuse.
