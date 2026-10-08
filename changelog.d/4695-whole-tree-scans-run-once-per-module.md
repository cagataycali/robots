### Tests: a grader walks the tree once per module, not once per cell

Twenty-one graders computed the same whole-tree scan (every package file
parsed and walked) inside each of their cells - the finiteness grader six
times, the private-state and mixed-return graders twice. The zero-argument scan
behind each is now `functools.cache`d, so a module pays for its walk once. Same
verdicts, same planted negatives; the 21 files run in 88 s instead of 163 s on
two cores.
