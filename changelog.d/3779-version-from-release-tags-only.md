### Fixed: the build reads its version from release tags only

`git describe` accepted any tag containing a digit, so a clone that also carries
a fork's artifact tags either failed `pip install -e .` outright with `Can't
parse version from tag 'artifacts-pc020-obstruction'`, or quietly took the tag's
trailing digits as the version (`artifact-bench-video-1783229263` built as
version `1783229263`). The describe command now matches `v[0-9]*`, so only
release tags are asked to carry a version.
