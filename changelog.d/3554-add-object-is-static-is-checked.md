### Fixed: `add_object`'s `is_static` is checked, not read by truthiness

`is_static` is tri-state (`True` / `False` / `None` for unspecified), and the
resolution that reads it tests by **identity** (`is_static is False`,
`is_static is None`) while every later read is a truthiness one. A supplied
value outside that domain escaped both, on all three backends. Measured against
a real compiled MuJoCo model, one `add_object` per case:

    add_object(shape="plane", is_static=False)  -> status="error"   (documented)
    add_object(shape="plane", is_static=0)      -> status="success", body static
    add_object(shape="plane", is_static="")     -> status="success", body static
    add_object(shape="plane", is_static=np.False_) -> status="success", body static

`0` *is* the value that refusal answers - `int(False) == 0` - so the same
request got two verdicts depending on how it was spelled, and the spelling that
was accepted got exactly the quiet override the refusal exists to prevent: the
caller is told a dynamic plane was built and gets a static one. `np.False_` is a
value this flag's own domain (`boolean_flag_error`) *accepts*, and it took the
same path, because `np.False_ is False` is `False`.

The truthy half inverted the other way:

    add_object(shape="box", is_static=False)   -> free body,  fell 0.4751 m in 400 steps
    add_object(shape="box", is_static="false") -> welded body, fell 0.0000 m, status="success"

`"false"`, `"no"` and `"off"` - the spellings an operator reaches for to opt out
- are non-empty strings, so each welded a body the caller asked to be dynamic,
and `'false'` was then stored on `SimObject.is_static`, which is annotated
`bool` and read by `list_objects`, the scene rebuild and domain randomization.
Newton and Isaac read the flag purely by truthiness, so `"false"` fixed a body
there too and `0` was stored verbatim.

A supplied `is_static` now goes through the shared posture domain
(`SimEngine._validate_posture_flags`) on MuJoCo, Newton and Isaac, so a spelling
one backend refuses is refused by all of them, and is normalized to a plain
`bool` so the identity reads below it are sound and no `numpy` scalar reaches
the record. `None` remains the documented "unspecified" sentinel and is
untouched; the plane refusal, its wording and its named remedy are unchanged,
and `True` / `False` / `None` / `np.True_` / `np.False_` all behave exactly as
documented.
