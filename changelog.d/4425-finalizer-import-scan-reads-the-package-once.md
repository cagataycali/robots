### Quality: the finalizer import scan reads the package once

The check that no `cleanup`/`destroy`/`__del__` imports a standard-library
module was one test cell per package file (372 cells), and every cell re-read
the whole package to look up its one file. It is now one test over a cached
read that still names every offender by `file:line`. The module ran 400 tests
in 32.8 s before and 29 in 11.2 s after, on one process.
