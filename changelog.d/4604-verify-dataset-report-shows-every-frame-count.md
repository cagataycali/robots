### Fixed: `verify-dataset` prints every frame count it read

The human report of `strands-robots verify-dataset` now always prints
`frames   (parquet)`, including when it is 0, and prints `info.json frames`
whenever the header declares it. Before, a dataset whose episodes held no frames
showed no frame line at all, and a header that claimed more frames than the
parquet held was visible only in the problem list. The `--json` output is unchanged.
