### Fixed

- **policies/vera/docker**: the offline checkpoint resolver skips a
  `provenance.json` it cannot read as a run record, instead of raising on it.
  `_index_local_ckpts` already skipped a file that does not parse and one that
  omits `wandb_run`, but handed anything else to the two operations that assume a
  string - using the value as a dict key and splitting it on `/` - so a payload
  that parses into an array, string, number or null, or an object whose
  `wandb_run` is a number, list or object, raised out of the scan. That is fatal
  rather than local: the module self-installs on import and `launch_server.py`
  imports it before the server module, so one malformed record among the mounted
  checkpoints killed the container entrypoint before the server was reached, and
  the whole root is scanned in one pass, so the healthy records went with it -
  including `37oa162u`, the run the entrypoint defaults to for mimicgen. The
  tolerance was backwards: a file that could not be read at all was skipped,
  while one that parsed into the wrong shape was fatal. Such a record is now
  skipped and named in a warning with the type that made it unusable, following
  the rule `transforms.provenance.load_provenance` applies to a provenance
  payload - check the type before using the value - and differing only in
  disposition, because this resolver has a fallback (the network) where a caller
  asking which episodes are synthetic has none. Every record indexed before is
  indexed on identical terms, and a checkpoint directory carrying no wandb run is
  still skipped silently, since that is the ordinary case for every artifact not
  loaded by run id.
