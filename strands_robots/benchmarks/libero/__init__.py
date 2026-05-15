"""LIBERO benchmark adapter - see :mod:`strands_robots.benchmarks.libero.adapter`.

Public surface (re-exported from submodules so agents can do
``from strands_robots.benchmarks.libero import LiberoAdapter``):

* :class:`LiberoAdapter` - ``BenchmarkProtocol`` built around a BDDL task.
* :func:`load_libero_suite` - bulk-register every task in a suite.
* :class:`BDDLParseError` - raised on malformed BDDL input.

The adapter and parser have **no** dependency on the ``libero`` pip
package - you can use them with your own BDDL files. Only
:func:`load_libero_suite` touches the upstream package (to discover task
files), and only when you don't pass an explicit ``bddl_dir=``.
"""

from strands_robots.benchmarks.libero.adapter import BDDLParseError, LiberoAdapter
from strands_robots.benchmarks.libero.bddl_parser import (
    PREDICATE_VOCABULARY,
    BDDLProblem,
    compile_goal,
    parse_bddl,
    parse_bddl_file,
)
from strands_robots.benchmarks.libero.suite import (
    SUITE_NAMES,
    available_suites,
    load_libero_suite,
)

__all__ = [
    "BDDLParseError",
    "BDDLProblem",
    "LiberoAdapter",
    "PREDICATE_VOCABULARY",
    "SUITE_NAMES",
    "available_suites",
    "compile_goal",
    "load_libero_suite",
    "parse_bddl",
    "parse_bddl_file",
]
