### Tests: the rclpy-refusal grader parses only the files that can hold an offender

`tests/test_rclpy_refusal_cells_establish_the_absence.py` reports a cell that
expects an `ImportError` from an rclpy-probing surface without establishing
the absence, and it parsed every test file to find one. The rule reads the
exception as a bare name, and an identifier appears verbatim in the source
that declares it, so a file without the substring holds no offender and is
no longer parsed - 1,711 of 1,896 files at the time of the change. The package
is parsed once for both derivations and the surface set once for both cells.
The cell was the largest in the suite and the last one running in both of the
measured CI logs, so the cut lands on the suite's wall clock rather than in
its average (#4043, towards #3869).
