"""Distribution is a property of the ``test`` script, not of the repository.

``hatch run test`` spreads the unit suite over the cores with ``-n logical
--dist loadfile``. Those flags could equally sit in ``[tool.pytest.ini_options].addopts``,
and that is where a first cut put them -- but ``addopts`` is read by every
pytest invocation rooted here, ``hatch run test-integ`` included, and the
integration suite is the one population that cannot run two files at once:
its cells bind fixed ports and named containers, hold the one GPU, and open
physical serial buses. Distributing it silently would have been a default
changed on a path nothing measured, with two writers on one arm as the failure.

So the split pinned here is: the ``test`` script asks for the distribution, and
nothing an invocation inherits does. A bare ``pytest`` and ``test-integ`` stay in
one process until someone asks for otherwise on the command line.
"""

from __future__ import annotations

import shlex
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: Every spelling that turns a session distributed. ``-n`` also comes glued to
#: its value (``-n4``), which is why the check reads prefixes below.
_DISTRIBUTING = ("-n", "--numprocesses", "--dist", "--tx")


def _distributing_tokens(command: str) -> list[str]:
    """Return the tokens of ``command`` that would distribute a session."""
    return [token for token in shlex.split(command) if token.startswith(_DISTRIBUTING)]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))


def test_the_test_script_distributes_one_worker_per_file(pyproject: dict) -> None:
    """The unit suite's own script carries the flags, so ``hatch run test`` is distributed."""
    script = pyproject["tool"]["hatch"]["envs"]["default"]["scripts"]["test"]
    tokens = shlex.split(script)
    assert tokens[:1] == ["pytest"], script
    assert "-n" in tokens and tokens[tokens.index("-n") + 1] == "logical", (
        f"the test script does not distribute over the logical CPUs ({script!r}). Without "
        "-n the required check runs 58,000 tests in one process, which is the 51-minute step "
        "distribution replaced; with -n auto it does the same on part of the ubuntu-latest pool, "
        "because xdist resolves auto through psutil to the physical core count and a "
        "one-core/two-thread runner then gets one worker (measured: 249 s against 118 s for "
        "logical, same 9,456 tests, same runner)"
    )
    assert "--dist" in tokens and tokens[tokens.index("--dist") + 1] == "loadfile", (
        f"the test script distributes without loadfile ({script!r}); the file-scoped fixtures "
        "and module-level state this suite has assume one file stays on one worker"
    )


def test_nothing_an_invocation_inherits_distributes_it(pyproject: dict) -> None:
    """``addopts`` reaches ``tests_integ/`` as well, so it must not carry the flags."""
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    assert _distributing_tokens(addopts) == [], (
        f"addopts distributes every pytest session rooted here ({addopts!r}), hatch run "
        "test-integ included; put the flags on the test script, where only the unit suite reads them"
    )


def test_the_integration_suite_stays_in_one_process(pyproject: dict) -> None:
    """Its files bind ports, containers, a GPU and serial buses one at a time."""
    script = pyproject["tool"]["hatch"]["envs"]["default"]["scripts"]["test-integ"]
    assert shlex.split(script)[:2] == ["pytest", "tests_integ/"], script
    assert _distributing_tokens(script) == [], (
        f"test-integ distributes ({script!r}); two of its files in flight at once is two "
        "writers on one port, one GPU or one physical arm"
    )
