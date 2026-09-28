### Deprecated: the `isaac` and `newton` backends move out of the package in 0.7

`create_simulation("isaac")` and `create_simulation("newton")` (and so
`Robot(..., sim="isaac")`) now raise a `DeprecationWarning` naming the
install line of the `strands-robots-sim-extras` plugin, which registers both
engines under the same backend names through the `strands_robots.backends`
entry-point group. Each backend's page carries the notice.

| moves out in 0.7 | install instead |
|---|---|
| `isaac` | `pip install 'strands-robots-sim-extras[isaac] @ git+https://github.com/cagataycali/strands-robots-sim-extras'` |
| `newton` | `pip install 'strands-robots-sim-extras[newton] @ git+https://github.com/cagataycali/strands-robots-sim-extras'` |

Nothing else changes in 0.6: both backends still construct and run in-tree.
