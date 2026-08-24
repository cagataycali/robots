### Quality: the MotionBricks tests pin that the instruction string is discarded, instead of only saying so

`MotionBricksPolicy.get_actions` takes an `instruction` and reads no goal from
it — the clip comes from the well-known `style` / `mode` / `locomotion_style`
kwargs. That is correct for a *style*-driven generative model, and four places
already said so: the policy's class docstring, `get_actions`' own docstring,
`docs/policies/motionbricks.md`, and that page's "style-driven" frontmatter.

Nothing pinned it. Every one of the twenty `get_actions_sync` calls in
`tests/policies/motionbricks/test_policy.py` passed `""` as the instruction, so
the claim held **vacuously**: the suite never supplied a non-empty one, and
wiring the instruction into the goal resolution would have turned all four prose
claims into lies with every test still green.

The gap matters because the discard is silent on the parameter a caller reaches
for first. `instruction="walk stealthily towards the door"` is accepted, steers
nothing, and returns a successful 29-joint action dict, so a caller's stated
intent is dropped with no signal. Refusing a non-empty instruction is not the
remedy — every `Policy` accepts one and the runner forwards it verbatim to the
providers that do read it — so the discard stays and a test keeps it honest.

The pin drives one tick per probe instruction and asserts the generator is fed
byte-identical control signals and returns an identical action dict. The probes
are chosen to be discriminating rather than inert: three are exact clip names the
stub generator really has, so a style resolution that consulted the instruction
would select a different mode. A companion test asserts exactly that — each of
those names resolves to a mode other than the pinned `walk` — so the pin cannot
pass merely because the probe strings were unresolvable.
