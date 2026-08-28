### Fixed: the Microduck policy checks the graph's action before it uses it

`MicroduckPolicy` holds three width contracts and checked two of them:
`default_pose` against `len(joint_names)` in `_ensure_config`, and a `command`
override against the width `command_names` declares in `_apply_command_kwargs`
(whose own comment says the width check "lives in one place"). The third - the
width of the action the ONNX graph itself returns - was not checked, and it is
the one that differs from the other two by being fed *back*: `last_action` **is**
that array, so it sets the width of the observation the graph is handed on the
next tick.

Measured with an injected stub session returning a width other than the joint
count, recording the observation width the graph is actually fed:

| graph action width | observation widths fed to the graph | outcome |
|---|---|---|
| 14 (the contract) | `[61, 61, 61]` | 3 x full 14-key action dict |
| 1 | `[61, 48, 48]` | 3 x full 14-key action dict |
| 15 / 13 / 7 | `[61]` | `ValueError` from inside numpy |

A width of 1 broadcasts against `default_pose` inside `decode_action`, so it
decoded silently, gave every joint the same target, and from tick 2 onward handed
the graph a 48-wide vector where that graph's own `observation_names` metadata
declares 61 - reporting a full action dict throughout. Any other width raised
`operands could not be broadcast together with shapes (14,) (15,)` from inside
numpy's decode, naming neither this policy nor the graph.

A non-finite component had both consequences at once. A single `nan` in the raw
action commanded **1 of 14** joints `nan` and fed that `nan` back into the next
observation; an all-`inf` action commanded **14 of 14**. Both reported success.
`EmpiricalNormalization` is fused into the exported graph, so nothing downstream
sanitises the vector - which is the same reason the sibling scene-construction
guards refuse a non-finite component, and `finite_vector_error`'s own docstring
names both harms (a bare `ValueError` from a numpy assignment, and a `nan`
propagated while reporting `success`).

Both are now refused at the seam where the graph's output enters the policy,
while the expected width and its source are still in hand, in the style of the
two checks the class already performs. The finiteness half consults the shared
`finite_vector_error` domain rather than a local `isfinite` loop.

Nothing that worked changes: the contract width still reaches every joint over
repeated ticks, a `(1, n)` row-shaped action is still squeezed and accepted, the
legacy twist-only 51-D command width still works, a large but finite action is
still accepted (the guard is not a magnitude bound), and a robot with a joint
count other than 14 still works because the expected width is derived from
`joint_names` rather than a literal.
