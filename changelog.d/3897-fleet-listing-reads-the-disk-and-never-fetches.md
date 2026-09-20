### Fixed: the dashboard's fleet listing reads the disk and never fetches an asset

`/api/fleet` reports every registry robot with a `model_local` flag, and its
module promised a read with no side effects. The flag came from
`resolve_model()`, whose asset-manager step downloads what is not on disk, so
one GET against a cold cache tried to fetch all 64 sim-capable robots - two
tries each - and cloned every upstream asset repository the registry names:
4.4 GB and 63 s on one machine, the 35.6 s test cell #3869 ranks among the
suite's slowest. `/api/robots/{name}` did the same for one robot.
`resolve_model()` now carries the keyword-only `allow_download` that
`resolve_model_path()` already honours and hands it to both of its tries; the
ladder and the downloading default are unchanged for every caller that loads a
model, and the two fleet routes ask with `allow_download=False`. Pinned by
three tests, two of them recording every download attempt a route makes under
an empty asset tree. Towards #3869.
