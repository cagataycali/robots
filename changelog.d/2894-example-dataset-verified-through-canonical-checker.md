### Fixed: a recording example verifies its dataset through the canonical checker

`examples/kimodo/kimodo_g1_dataset_headcam.py` records N episodes and then confirmed the
count by re-reading the parquet itself, opening `data/chunk-000/file-000.parquet` by name.
LeRobot v3 does not guarantee that name. `data/` is written in files capped at
`DEFAULT_DATA_FILE_SIZE_IN_MB` (100 MB), so a recording that outgrows the cap spills into
`file-001.parquet` while the far smaller `meta/episodes` parquet still holds every row.
Reading only the leading file then sees only the leading episodes.

Measured on a healthy twenty-episode dataset whose `data/` spilled at episode twelve:
`verify_dataset` reported `status='success'` with `total_episodes=20`, and the example's own
check raised `parquet truth FAIL: unique episode_index=[0, ... 11], expected [0, ... 19]`.
A head-cam dataset -- two camera streams, twenty episodes -- passes 100 MB well before it
finishes, so the check written to catch the mega-episode corruption (every frame buffered
into `episode_index=0`) was refusing the successful runs instead. Its own docstring calls
that check "the same contract the autonomous harness uses", which is what makes the false
refusal expensive: it trains a reader to disbelieve the one signal that would catch a real
fabrication.

The example now calls `strands_robots.verify_dataset.verify_dataset(root, expected=N)`,
which globs `meta/episodes/**/*.parquet` and takes that as the ground truth, so it is
size-independent by construction. It also checks two corruptions a count-only read cannot
see: a per-episode MP4 that is absent, empty or truncated, and an `action` /
`observation.state` column written identically zero across an episode. The head-cam feature
assertion stays in the example, because that both cameras are declared is this script's own
contract rather than a property every LeRobot dataset has to satisfy.

This does not widen what is refused. `meta/episodes` is the declared ground truth, and
`verify_dataset` already reported success for a dataset whose `data/` tail is unreadable --
it did so before this pairing too. The change removes a false refusal; it does not add
detection of a corrupt `data/` file.
