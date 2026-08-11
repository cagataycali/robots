# start_recording: reporting an unavailable LeRobot dataset stack

Measurement behind the PR that drives all nine cells.

* `census.py`  - the coverage census (per-function views) that found the block.
* `capture.py` - measures the nine diagnoses on the three backends, then records
                 one real dataset end to end and reads a frame back out of it.
* `compose.py` - builds the figure; every rendered number is asserted against
                 `facts.json` before the image is written.
* `facts.json` - the raw measurement.
* `mutate.py`  - the mutation table (7 regressions x 2 test sets).
