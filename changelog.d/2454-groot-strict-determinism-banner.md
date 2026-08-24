### Fixed

- **policies/groot**: the determinism wrapper's startup banner now reports the mode the server
  ended up in rather than the one it was asked for. Enabling strict mode is best-effort - an op
  with no deterministic kernel makes `torch.use_deterministic_algorithms` refuse, which degrades
  the server to non-strict instead of killing it - but the banner read the request, so such a run
  logged `strict=True` one line after its own failure warning. That banner is the only record of
  a container run's determinism configuration, so an investigation into a rollout that failed to
  reproduce would read it and rule out the one setting that was in fact missing. A refused
  request now reads as off and names itself.
