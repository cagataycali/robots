### Tests: a shared source tree is walked once per process, not once per grader

The 221 graders that walk the shared parse of the package or of `tests/`
each ran `ast.walk` over it again. `tests._package_ast.walk_tree` lists a
shared tree's nodes on the first walk and replays them, in the same order,
after that; any other node is walked as before. Same verdicts, same planted
negatives; the 183 whole-tree graders run in 309 s instead of 421 s on two
cores.
