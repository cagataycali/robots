"""
Repro: strands_robots is missing the standard ``__version__`` attribute.

Fresh install (``pip install strands-robots``, currently 0.5.2), Python 3.12:

    >>> import strands_robots
    >>> strands_robots.__version__          # standard Python convention (PEP 396)
    AttributeError: module \'strands_robots\' has no attribute \'__version__\'

Root cause:  strands_robots/__init__.py defines a ``__getattr__`` fallback that
raises AttributeError for anything not in ``_LAZY_IMPORTS`` (see lines 268-289),
and ``__version__`` is never registered. ``pyproject.toml`` uses
``dynamic = ["version"]`` with hatch-vcs, so metadata knows the version
(``importlib.metadata.version("strands-robots")`` returns "0.5.2"), but the
module itself does not expose it.

Impact (real end-user workflow):

- Users bug-reporting strands-robots are asked to share
  ``strands_robots.__version__`` -- they get AttributeError.
- Diagnostic / doctor tools that walk installed packages and read
  ``mod.__version__`` (numpy, torch, transformers, mujoco, lerobot ALL support
  this) report strands-robots as "unknown".
- Refs closed harness issue #66 which fixed *doctor\'s* own reporting via
  importlib.metadata -- the underlying gap in the package remained.

Expected: ``strands_robots.__version__`` returns the same string as
``importlib.metadata.version("strands-robots")``.

Fix (one-liner) in strands_robots/__init__.py near the other _importlib helpers:

    from importlib.metadata import version as _pkg_version
    from importlib.metadata import PackageNotFoundError as _PkgNotFound
    try:
        __version__ = _pkg_version("strands-robots")
    except _PkgNotFound:  # editable install without dist-info
        __version__ = "0.0.0+unknown"

Run:  python bugbash/missing_dunder_version_repro.py
Exits 1 with "REPRO OK" on defect; exits 0 (with printed version) when fixed.
"""
import importlib.metadata as m
import sys

import strands_robots

print("metadata version:", m.version("strands-robots"))
try:
    v = strands_robots.__version__
    print("strands_robots.__version__:", v)
    sys.exit(0)
except AttributeError as e:
    print(f"REPRO OK -- AttributeError: {e}")
    sys.exit(1)
