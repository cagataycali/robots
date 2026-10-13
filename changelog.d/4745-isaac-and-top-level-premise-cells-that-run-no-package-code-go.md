### Tests: Isaac, inference and top-level suites drop the cells that ran no package code

37 cells across 15 suites under `tests/`, `tests/inference/` and
`tests/simulation/isaac/` asserted what the standard library or numpy does
with a value the package already refuses: `threading.Event.wait`,
`Thread.join`, `ThreadPoolExecutor` workers, `importlib` reimports and
namespace packages, `range`, `min`/`max` with `nan`, `len` on 0-d arrays,
`np.random.default_rng`, `dict.get` and `math.tan`. Coverage contexts show
each of them executed zero lines of `strands_robots`, so they go. Cells that
read the package's own source or constants, use MuJoCo, lerobot, zmq or torch
as an oracle, or guard a fixture against being vacuous stay. The 15 files go
from 806 collected cells to 769.
