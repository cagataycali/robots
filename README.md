# artifacts/recording-posture-flag-domain

`capture.py` is run unchanged in two checkouts - `upstream/main` and the PR branch -
and each run records its own import tree into `facts.json`, so the two halves are
provably from different trees.

It records one episode with `overwrite=False`, then re-opens the dataset to append a
longer second episode while opting out of overwrite as `overwrite="false"`, and reads
`meta/info.json` before and after. On the PR branch it then follows the advice the
refusal gives (`overwrite=False`) so the remedy is verified rather than asserted.

- `facts-main.json`   -- both calls `success`; 1 episode / 8 frames becomes 1 / 12 (the
  recorded episode was deleted).
- `facts-branch.json` -- refused, dataset still 1 / 8, then `overwrite=False` appends to
  2 episodes / 20 frames.

The MuJoCo render is byte-comparable across the two trees (max |delta| = 1/255,
renderer noise), and the frame strip is decoded back out of the dataset's own MP4.
