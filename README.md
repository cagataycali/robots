# Isaac base-pose readback: every documented route driven

`capture.py` traces each "could not be read" route to the `return None` arm it
reaches and marks the arms the base coverage run reported missing, then renders
what the readback's docstring warns about ("a wrong base makes every world-frame
target silently wrong") on the MuJoCo backend, since Isaac Sim is not installed
on this host.

`mutations.py` is the mutation table: it scopes each anchor to
`_articulation_base_pose`'s own line range (the sibling joint read carries a
byte-identical exception arm, so a file-wide replace would mutate the wrong
function) and runs both arms.

`compose.py` asserts every drawn number against `facts.json` before saving.
