### Fixed: cross-repo references in the Isaac backend name the repository they belong to

`strands_robots/simulation/isaac/` was absorbed from `strands-labs/robots-sim` by
#1156 and arrived carrying that repository's issue and pull-request numbering in
the bare `PR #N` / `issue #N` form, which resolves against *this* repository. All
18 such references are now written `robots-sim#N`.

The two that could be checked cheaply were both wrong, in the two different ways
that matter:

| reference | means in robots-sim | resolves here to |
|---|---|---|
| `PR #117` | fail fast on the LIBERO `load_scene` gap | **nothing** - this repo has no #117 |
| `PR #31` | `IsaacSimulation` backend skeleton (R7) | #31 `chore: code hygiene, logging cleanup, and f-string logging migration` |

`PR #117` fails loudly, which is the harmless case. `PR #31` is the expensive
one: the docstring calls it "the exception-hygiene pin", and this repository's
#31 is a real merged *code hygiene* PR -- so a reader who follows the reference
to check the claim finds a plausible-sounding match and concludes it verifies. A
false pass is worse than a dead link, because nothing prompts a second look.
`issue #69` and `issue #88` behave the same way: both numbers exist here, both
are unrelated CI issues.

Comment and docstring text only -- no behaviour change. The two sites that
already carried a full URL keep it; only their misleading bare link text
changed. Three genuine local references in the package (`issue #1537` twice,
`issue #1812`) are deliberately left bare, because they are correct.

`tests/simulation/isaac/test_migrated_reference_provenance.py` pins the rule. It
is scoped to this package -- the only part of the tree holding two numbering
namespaces in one syntax, since `strands_robots/` elsewhere cites this repo's own
`PR #85` / `PR #92` / `PR #86` bare and correctly -- and thresholded at 1000
rather than absolute: robots-sim never issued a number above 173, while this
repository was already at #1156 when it absorbed the backend, so the ranges
cannot overlap and a four-digit reference here is unambiguously local.
