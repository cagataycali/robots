### Tests: nine whole-tree sweeps grade the tree in one cell, not one cell per file

Nine graders that check every package module, example script, agent tool,
`Args:`/`Attributes:` surface or mesh prose file used to parametrize over the
files they read, so each file was its own test cell: 1,744 cells in all.
Each sweep now walks the same population in one cell and lists every
offender in one message, so a failure still names each file and line. The
nine files go from 1,744 collected cells to 100. The planted-defect cells stay,
and two sweeps (`examples/` lerobot imports, sim examples) now also fail when
they find nothing to grade, where an empty population used to pass as a skip.
