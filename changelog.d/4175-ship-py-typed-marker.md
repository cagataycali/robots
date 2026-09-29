### Fixed: the package ships a `py.typed` marker, so type checkers read its annotations

The wheel carried no PEP 561 marker, so a type checker treated every import
from `strands_robots` as `Any` even though the package annotates its public
surface and checks most of its modules under `disallow_untyped_defs`. A
downstream project running `mypy --strict` got `import-untyped` errors for
`SimEngine`, `Policy` and `HardwareDriver`, and a subclass of any of them went
unchecked: an override with a wrong return type or an incompatible signature
passed silently. `strands_robots/py.typed` now ships at the package root as an
empty marker (the annotations are inline, so it is not a `partial` stub
package), and the project declares the `Typing :: Typed` classifier. A project
that silenced the import with `ignore_missing_imports` now sees the real
annotations, so previously hidden type errors can surface, and under `--strict`
it may see `no-untyped-call` from the modules its own mypy configuration relaxes.
