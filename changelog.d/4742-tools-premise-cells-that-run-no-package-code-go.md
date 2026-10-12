### Tests: tool domain suites drop the cells that ran no package code

35 cells across six `tests/tools/` suites asserted what `accelerate`, `torch`,
`roslibpy`, `autobahn`, lerobot's config fields or plain Python comparisons do
with a value the tool already refuses (`not nan > 1`, `DataLoader(batch_size=0)`).
Coverage contexts show each of them executed zero lines of `strands_robots`, so
they go; the reasons they recorded already live on the gates' own comments.
Cells that read the package's own source, or drive package code, stay. The six
files go from 174 cells to 139.
