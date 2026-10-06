### Docs: robot pages mark a body-bound provider that is removed in 0.7

The G1 page and the catalog's coverage matrix listed `kimodo` and
`protomotions` next to `wbc`, `holosoma` and `wbc_gait` with nothing to say
they are removed in 0.7, while `create_policy` warns on both and their own
pages carry the banner. The generated lists now print `(removed in 0.7)` after
every provider in `strands_robots.policies.factory._REMOVED_IN_0_7`, read
from the source, so the mark follows the factory when a provider is added or
cut.
