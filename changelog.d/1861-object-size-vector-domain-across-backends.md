### Fixed: an `add_object` size vector is validated on every backend

`add_object(size=...)` reached the Newton and Isaac backends unvalidated, while
the MuJoCo backend had refused the same values since its numeric inputs were
hardened. Measured with one `add_object` per case and no `newton` / `warp` /
`isaacsim` installed, eight values MuJoCo refuses were accepted by at least one
of the other two: an empty vector, a `nan` or `inf` component, a `bool`, a
`None`, a nested list, the bare string `"abc"`, and a scalar.

Newton read the vector for truthiness (`size or default_size`) - the last
surviving coalesce of that shape on that constructor, where the other four vector
parameters all test `is None`. So `np.array([0.1, 0.1, 0.1])`, which every
docstring there advertises as accepted, raised
`ValueError: truth value of an array ... is ambiguous` straight through the
structured envelope these methods document as their only failure channel, and
`[]` is falsy so it read as *omitted*: the default 5 cm extent was applied and
the call reported success. Everything else was stored on the registry entry and
reached the solver at rebuild time rather than at the `add_object` a caller can
attribute it to - `"abc"` raised
`TypeError: can only concatenate str (not "list") to str` out of the box branch,
and built a sphere of `radius='a'` out of the sphere one.

Isaac coerced with `list(size)`, which validated nothing: the same non-finite,
`bool` and `None` components reached the prim constructor, `"abc"` was split per
character into the 3-component extent `['a', 'b', 'c']`, and a scalar raised
`TypeError: 'float' object is not iterable` out of that `list()` call, past the
envelope. A NumPy extent also survived into the agent-visible result `json` as
`[np.float64(0.1), ...]`.

All three now share one definition, `strands_robots.utils.coerce_size_vector`,
composed of the same `finite_vector_error` component domain MuJoCo already used -
so the refusal is identical word for word, not merely identical in verdict - plus
the rule that an empty vector is a component count rather than an omission.
NumPy input is accepted and normalized to plain floats, and a refused size
constructs no prim, registers no object and leaves the name reusable.

Three axes are deliberately unchanged, because each depends on the shape and
needs one contract decision rather than a helper default: the per-shape component
count, whether a short vector may be completed from trailing defaults (the Isaac
`size` docstring promises this, MuJoCo refuses it outright), and whether a
consumed extent must be positive. #1858 tracks them, and the boundary is pinned
rather than left implicit.
