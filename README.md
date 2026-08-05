# StreamingDatasetReader.open numeric domains - measurement

`measure_stream.py` opens a locally cached `lerobot/pusht` (206 episodes,
25 650 frames, proprio-only so no torchcodec is needed) once per case and
records what `open()` returned, the shard count the dataset ended up with, and
how many frames actually came out.

Run once per tree; each dump records the tree it measured.

    python3 measure_stream.py out.json

* `before_upstream_main.json` - measured on `upstream/main`
* `after_this_change.json` - measured on the branch
* `streaming_domains.png` - composed from the two dumps
