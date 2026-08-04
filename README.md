# reward-config asset failure: measurements

Every number in the PR body and figure is produced by these scripts.

- `premise_probe.py` - the premises the fix rests on: every `huggingface_hub`
  failure class for an unobtainable file is an `OSError` subclass, and
  `try_to_load_from_cache` is a pure local lookup.
- `remedy_probe.py` - verifies the remedy the error message names: with
  `vlm_config` supplied, construction reaches neither `AutoConfig.from_pretrained`
  nor `AutoTokenizer.from_pretrained` (both made fatal).
- `measure_facts.py` - run once per tree with a cold `HF_HOME` and
  `HF_HUB_OFFLINE=1`; each run records its own tree path.
- `make_figure.py` - composes the figure and asserts every claim against the two
  JSON dumps before saving.
- `facts-upstream-main.json` / `facts-this-change.json` - the raw measurements.
