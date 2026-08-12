### Fixed: report which tensors a too-old diffusers could not fill for a Cosmos 3 checkpoint

`Cosmos3Policy(backend="diffusers")` now refuses a checkpoint the installed
`diffusers` cannot build, instead of letting it through on randomly initialized
weights. `Cosmos3OmniPipeline.from_pretrained` does not raise on an architecture
mismatch - it logs "newly initialized" warnings and leaves the unmatched
parameters on the `meta` device - so with `nvidia/Cosmos3-Edge` (built against
diffusers 0.40.0.dev0) on diffusers 0.39.0, 112 of 633 transformer parameters
were left unfilled and the only symptom was a bare
`NotImplementedError: Cannot copy out of meta tensor` from the following device
copy, naming neither diffusers, nor its version, nor the checkpoint. The
backend's one actionable remedy was reachable only through `ImportError`, which
that state never raises. The load now reports the checkpoint, the installed
diffusers, the unfilled tensors and the upgrade command, and reports it before
the device copy - so the silent case, where the copy happens to succeed, is
caught too.

Packaging on the same axis. `Cosmos3OmniPipeline` and `CosmosActionCondition`
first ship in diffusers 0.39.0 (0.36.0, 0.37.1 and 0.38.0 carry neither), so:

- the `cosmos3-diffusers` extra's floor moves `>=0.30` -> `>=0.39`;
- the `[tool.uv]` diffusers override moves `>=0.38.0` -> `>=0.39`. A uv override
  *replaces* a requirement rather than intersecting with it, so at `>=0.38.0` it
  silently discarded the extra's floor and `uv.lock` pinned diffusers 0.38.0 - a
  release carrying no `Cosmos3OmniPipeline` at all. 0.39 clears both the CVE
  floor the override exists for (fixed in 0.38.0) and the capability floor;
- `uv.lock` now resolves diffusers 0.39.0;
- the install hint no longer claims the pipeline is source-only.

`nvidia/Cosmos3-Nano` loads cleanly on diffusers 0.39.0 and 0.40.0.dev0 alike, so
the required diffusers is a property of the checkpoint rather than of the library:
that is why a version range cannot express it and the load reports it instead.
