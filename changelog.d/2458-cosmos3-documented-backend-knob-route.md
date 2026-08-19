### Fixed

- **policies/cosmos3**: the safety-checker note in `docs/policies/cosmos3.md` told readers to
  `pass enable_safety_checker=True`, which no surface the page names accepts - `Cosmos3Policy`
  declares no such parameter and takes no `**kwargs`, so both documented routes raise `TypeError`,
  and the registry route filters the key out and runs with the checker off. The note now names the
  `Cosmos3DiffusersBackend` parameter it is, with a snippet handing that backend to
  `Cosmos3Policy(diffusers_backend=...)`, and lists the other load and sampling knobs the same
  route carries. `Cosmos3Policy`'s own `diffusers_backend` docstring no longer calls itself
  test-only injection.
