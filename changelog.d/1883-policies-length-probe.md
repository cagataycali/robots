### Fixed: a 0-d observation value is read as the scalar it is, not raised on

`MockPolicy.get_actions` raised `TypeError: len() of unsized object` when
`observation.state` was a 0-d array (`np.array(0.5)`, or the result of a
reduction such as `np.mean(...)`), and `WBCPolicy._read_vec` raised the same for
a 0-d `base_ang_vel` / `base_quat` entry, mid-rollout. Both already had an
answer for a value that carries no component count - `MockPolicy` falls back to
six joints, `_read_vec` returns `None` - and a 0-d array was the one spelling
that never reached it, because `hasattr(value, "__len__")` is `True` for a value
whose `__len__` raises.

The three remaining sites of that idiom now read the count through
`utils.sequence_length`, which has been the single owner of the rule since it
answered the same question for the simulation and rendering surfaces. The third
site, `WBCPolicy`'s flat-vector/per-joint observation discriminator, did not
raise but split the two spellings of a scalar: a plain `float` was consumed as
the per-joint form while a 0-d array took the flat form and wrote its single
value into the first joint's slot. Both are now read as the per-joint form.

Correctly sized states and observation vectors are accepted exactly as before,
and the structural guard that keeps the idiom out now scans the whole package
rather than the two subpackages named when it was written - which is why these
three survived.
