### Fixed: a VERA camera selection that names no view is refused, not served every view

`VeraPolicy.image_keys` names a SUBSET of the observation's own image keys - the
cameras to width-concat into the single `(H, W, 3)` frame the video planner acts
on - and `None` is documented as "the server's `view_keys`, or every image key in
the observation". It was read by truthiness at both the site that stored it and
the site that resolved it, so an **empty** selection took that same branch: a
caller who excluded every camera drove the arm from all of them, with no error
and no log line.

Measured against a two-camera observation, `image_keys=[]` and `image_keys=""`
were byte-identical to `image_keys=None` at both surfaces - stored as `None`,
resolved to both cameras - under a server that advertised `view_keys` and under
one that did not. The one refusal that could have caught it, `_extract_frame`'s
"requires at least one camera frame", was unreachable for an empty selection
because the selection had already been widened.

`image_keys` is now read `is not None`, and an empty selection is refused in the
constructor - before any server or config work, so a refused selection leaves
nothing to undo. The list shape stays with the shared `name_list_error` domain,
which is where `""` is answered as the bare string it is; only the emptiness
verdict is local, which is what that domain reserves for the caller.

`action_mapping` on the next line reads the same way and is unchanged: it is a
rename mapping rather than a subset of a collection the call owns, so `{}` and
`None` both honestly mean "no rename, columns keep their server names".

Pinned by `tests/policies/vera/test_vera_image_keys_selection_domain.py`.
