### Docs: the Isaac backend guide is 1,492 words, with parity on its own page

`docs/simulation/isaac.md` was 4,725 words - install and `IsaacConfig` in front
of a 2,085-word parity section, with the install narrative, the deleted
procedural builder, the `replicate()` stub and five "previously it did X"
passages in between. The history is what the changelog is for, and the contracts
are reference material a caller reads as one.

The history and the rationale are gone, the option lists and the
backend-specific verdicts are tables (`add_robot` sources, floating base,
meshes, cameras, loaders, scenes; the `name`, `mass=0`, unknown-entity and
`render` verdicts; the pip-install collateral as symptom, cause and remedy), and
the page is split at its H2: the guide keeps install, configuration and the two
runtime caveats - PhysX on the CPU, and the `reset()` a dynamic body needs - at
1,492 words, while the new `docs/simulation/isaac-parity.md` carries the
`SimEngine` parity, the refusal domains and `replicate()` at 1,112. No fact is
dropped, and `simulation/isaac.md` leaves the word-budget exemption list, so the
ratchet grades it from now on.
