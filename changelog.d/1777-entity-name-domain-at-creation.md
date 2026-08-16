### Fixed: a creation site refuses a name that cannot address the entity it creates

`add_object` / `add_camera` / `add_robot` claim a name, and three values were
accepted that produce an entity the rest of the same API cannot then address.
An empty name is MuJoCo's own sentinel for *unnamed*, so `add_object("")`
reported success and `get_body_state(body_name="")` then reported the body
absent - and because `render` routes `camera_name` in `(None, "", "default",
"free")` to the free camera, a camera created as `""` could never be rendered
from. A name containing a NUL registered one string while the model compiled
under another, so `mj_name2id` resolved the registry key and its truncation to
the same entity; through `add_robot` the NUL took the namespace separator with
it. A non-string name was entered in the registry and only then raised
`TypeError` out of the spec build, escaping the agent-tool dict these methods
document as their only failure channel and leaving the world holding an entry
for a body that does not exist, while any *falsy* non-string (`0`, `[]`) fell
into `add_robot`'s derive-a-label branch and reported success under a label the
caller never asked for.

All six creation sites (MuJoCo and Newton) now refuse those three through one
shared `entity_name_error`, before anything is registered, so the two backends
cannot drift. `add_robot(name=None)` and `add_robot(name="")` keep deriving a
label from the model on the MuJoCo backend - that short form is documented -
and nothing else narrows: a namespaced, dotted or non-ASCII name is
addressable and stays accepted. This is the creation-site half of the total
entity-name lookup, which resolves an unusable name to "absent" rather than
refusing it.
