### Fixed: `strands_robots.__version__` reports the installed version

`import strands_robots; strands_robots.__version__` raised `AttributeError`
from the package's lazy `__getattr__`; the version only lived in the
distribution metadata. The package root now reads it once from
`importlib.metadata.version("strands-robots")`, falling back to
`"0.0.0+unknown"` on a source tree that was never installed.
