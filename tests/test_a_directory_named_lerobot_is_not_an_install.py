"""A directory named ``lerobot`` on the import path is not a lerobot install.

``import lerobot`` succeeds for two states that mean opposite things. One is an
install. The other is any directory called ``lerobot`` that Python can see: it
imports as a namespace package, whose ``__file__`` is ``None`` and whose
``__path__`` is that directory, so every attribute and registry lookup through
it finds nothing. Three surfaces read ``import lerobot`` as the whole question
and reported the second state as the first:

- :func:`strands_robots.doctor.check_lerobot` printed ``PASS  lerobot ?`` - the
  install doctor attesting an install on a host with none, the unresolvable
  version the only tell.
- the ``use_lerobot`` discovery path returned a catalog of that directory's
  files under ``Modules:`` with ``robots (0)``, ``teleoperators (0)``,
  ``cameras (0)`` and ``policies (0)`` - which reads as "this install exposes
  nothing", not "there is no install", and is the answer an agent is handed.
- :func:`strands_robots.utils.ensure_lerobot_family_registered` logged
  ``lerobot is installed but lerobot.robots is not importable (partial
  install?)``, diagnosing an absent install as a damaged one.

Both spellings of the state are ordinary rather than contrived. A ``git clone``
of lerobot in the working directory is one (its sources live under ``src/``, so
the checkout root is not itself a package), and this repository's own
``examples/lerobot/`` is another: ``python examples/08_discover_lerobot.py``
puts ``examples/`` on ``sys.path`` as the script's directory, so the example
whose subject is "what does this LeRobot install expose?" answered with its own
sibling example scripts whenever the ``lerobot`` extra was missing.

The reachability question now has one owner,
:func:`~strands_robots.utils.lerobot_install_error`, and this module pins that
the three readers agree with it in all three states. ``installed`` is a
stand-in module carrying a ``__file__`` rather than the real package, so the
cells grade the rule on a host with no lerobot too.
"""

import subprocess
import sys
from types import ModuleType

import pytest

from strands_robots.doctor import check_lerobot
from strands_robots.tools.use_lerobot import use_lerobot
from strands_robots.utils import lerobot_install_error

#: The three states ``import lerobot`` cannot tell apart on its own.
ABSENT, SHADOWED, INSTALLED = "absent", "shadowed", "installed"


def _tool_text(result: dict) -> str:
    """Return the text a caller of the ``use_lerobot`` tool reads."""
    return "".join(block.get("text", "") for block in result.get("content", []) if isinstance(block, dict))


@pytest.fixture
def lerobot_state(request, monkeypatch, tmp_path):
    """Put ``import lerobot`` into one of the three states and name the directory.

    ``None`` in ``sys.modules`` is CPython's own "this import is blocked" entry,
    so ``absent`` raises the same ``ImportError`` an uninstalled host raises.
    """
    directory = tmp_path / "lerobot"
    directory.mkdir()
    if request.param == ABSENT:
        monkeypatch.setitem(sys.modules, "lerobot", None)
    else:
        module = ModuleType("lerobot")
        module.__path__ = [str(directory)]
        if request.param == INSTALLED:
            module.__file__ = str(directory / "__init__.py")
        monkeypatch.setitem(sys.modules, "lerobot", module)
    return directory


def test_the_import_machinery_really_reads_a_bare_directory_as_an_empty_lerobot(tmp_path):
    """The premise: a directory alone imports, with no file and no contents.

    Run with ``-S`` so site-packages cannot supply a real lerobot - a regular
    package wins over a namespace portion wherever it sits on the path, which is
    also why an installed host is unaffected by the shadow.
    """
    (tmp_path / "lerobot").mkdir()
    probe = "import lerobot; print(repr(getattr(lerobot, '__file__', None)), list(lerobot.__path__))"
    done = subprocess.run(
        [sys.executable, "-S", "-c", probe],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"},
    )
    assert done.returncode == 0, f"import failed: {done.stderr}"
    assert done.stdout.strip() == f"None ['{tmp_path / 'lerobot'}']", done.stdout


@pytest.mark.parametrize("lerobot_state", [ABSENT, SHADOWED, INSTALLED], indirect=True)
def test_the_doctor_reports_the_state_its_owner_reports(lerobot_state, request):
    """The install check and the owner agree, and a shadow names its directory."""
    state = request.node.callspec.params["lerobot_state"]
    problem = lerobot_install_error()
    doctor_line = check_lerobot()

    if state == INSTALLED:
        assert problem is None
        assert "PASS" in doctor_line, doctor_line
        return

    assert problem is not None and problem.startswith("lerobot not installed"), problem
    assert "PASS" not in doctor_line and "not installed" in doctor_line, doctor_line

    if state == SHADOWED:
        # The directory is the whole diagnosis: nothing else tells a reader why an
        # import that succeeded is not an install.
        assert str(lerobot_state) in problem and str(lerobot_state) in doctor_line, (problem, doctor_line)
        assert "namespace package" in problem, problem
        assert f"uv pip install -e {lerobot_state}" in problem, problem


@pytest.mark.parametrize("lerobot_state", [ABSENT, SHADOWED, INSTALLED], indirect=True)
def test_discovery_renders_a_catalog_only_for_an_install(lerobot_state, request):
    """A registry census is a claim about an install, so no other state renders one."""
    state = request.node.callspec.params["lerobot_state"]
    discovery = _tool_text(use_lerobot.__wrapped__(module="__discovery__", method="list_modules"))

    if state == INSTALLED:
        assert "not installed" not in discovery, discovery
        return

    assert "not installed" in discovery, discovery
    assert "robots (" not in discovery, discovery
    if state == SHADOWED:
        assert str(lerobot_state) in discovery, discovery


@pytest.mark.parametrize("lerobot_state", [ABSENT, SHADOWED], indirect=True)
def test_no_reachable_install_is_not_a_partial_install(lerobot_state, caplog):
    """The family walk keeps its "partial install?" warning for real damage."""
    from strands_robots import utils

    utils.ensure_lerobot_family_registered.cache_clear()
    with caplog.at_level("WARNING"):
        utils.ensure_lerobot_family_registered("robots")
    utils.ensure_lerobot_family_registered.cache_clear()
    assert "partial install" not in caplog.text, caplog.text
