### Changed: one grader enforces the module-not-filename citation rule across the whole package

Nine near-identical per-package copies of the docstring cross-reference guard are
replaced by a single table-driven grader over the whole tree. Resolution is now by
path rather than by bare name, so a citation of another project's file is left
alone even when its last component collides with an internal module name, and the
rule reaches the packages the per-package copies never covered.
