### Fixed: `strands-robots doctor --help` no longer runs the doctor

`doctor` had no argument parser, so `--help`, `--json` or a typo were ignored
and the full check ran under them. It now parses first: `-h` prints usage,
`--list` prints the check names without probing, and an unknown argument exits 2
with the usage line.
