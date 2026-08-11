"""The probe that decides which lerobot install remedy is printed, driven for real.

:func:`strands_robots.dataset_recorder._describe_lerobot_import_failure` chooses
between four different install instructions, and the first branch it takes is
``if not _lerobot_installed()``. That probe therefore decides whether a caller is
told to install lerobot or told that lerobot is already present and something
else is wrong -- the distinction the diagnosis exists to draw.

Nothing drove the probe. Every one of its five references replaced it with a
lambda, for a reason its own fixture states: the diagnosis tests stub it "so the
contract holds everywhere rather than only where a real lerobot happens to be
importable". That is the right call for tests whose subject is the *message*, and
it leaves the probe's own body unexercised -- so this module's subject is the
probe.

What the suite ran and what a caller runs were not the same branch. The probe
answers from ``sys.modules`` first and falls back to a spec lookup; importing
:mod:`strands_robots.dataset_recorder` does not import lerobot, so in a fresh
interpreter ``"lerobot" in sys.modules`` is False and the spec lookup is the
branch that answers. Under pytest something has already imported lerobot, so the
fast path answered every time and the lookup, its ``except`` and the absent
answer never ran.

The probe's docstring makes three claims, one per class below: it answers via a
spec lookup, it has no side effects (it must not pay lerobot's import cost just
to pick an error message), and -- because it is called from an error path -- a
lookup that raises has to be answered rather than propagated.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

from strands_robots import dataset_recorder as dr


def _lerobot_module_keys() -> list[str]:
    """Every live ``sys.modules`` key for lerobot or one of its submodules."""
    return [k for k in list(sys.modules) if k == "lerobot" or k.startswith("lerobot.")]


def _force_spec_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop lerobot from ``sys.modules`` so the probe reaches its spec lookup.

    ``monkeypatch.delitem`` restores each key at teardown, so the purge cannot
    leak into a later test that expects lerobot to be imported.
    """
    for key in _lerobot_module_keys():
        monkeypatch.delitem(sys.modules, key, raising=False)


def _lerobot_resolvable() -> bool:
    """Whether lerobot is installed, asked without importing it."""
    try:
        return importlib.util.find_spec("lerobot") is not None
    except (ImportError, ValueError):
        return False


def _patch_find_spec(monkeypatch: pytest.MonkeyPatch, outcome: Callable[[str], Any]) -> None:
    """Replace the spec lookup so each of its three outcomes can be driven."""

    def _lookup(name: str, package: str | None = None) -> Any:
        return outcome(name)

    monkeypatch.setattr(importlib.util, "find_spec", _lookup)


def _raises(exc: BaseException) -> Callable[[str], Any]:
    """A spec lookup that fails with ``exc``."""

    def _outcome(_name: str) -> Any:
        raise exc

    return _outcome


class TestThePathAProductionCallerTakes:
    """The spec lookup answers, and answering costs no import."""

    def test_the_module_under_test_does_not_import_lerobot(self) -> None:
        """The premise: the fast path is not the one a fresh caller reaches.

        Asked in a child interpreter because the pytest session has long since
        imported lerobot for other reasons; the question is what a process that
        imported only the recorder sees.
        """
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import strands_robots.dataset_recorder, sys; print('lerobot' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert proc.stdout.strip() == "False"

    @pytest.mark.skipif(not _lerobot_resolvable(), reason="lerobot is not installed")
    def test_the_spec_lookup_answers_true_for_an_installed_lerobot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_spec_lookup(monkeypatch)
        assert not _lerobot_module_keys(), "the purge must leave the fast path unreachable"

        assert dr._lerobot_installed() is True

    @pytest.mark.skipif(not _lerobot_resolvable(), reason="lerobot is not installed")
    def test_answering_does_not_import_lerobot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The docstring's stated reason for a spec lookup over an import.

        A question about which error message to print must not pay lerobot's
        import cost, so no lerobot module may appear in ``sys.modules`` because
        the probe was asked.
        """
        _force_spec_lookup(monkeypatch)

        assert dr._lerobot_installed() is True

        assert _lerobot_module_keys() == []


class TestTheFastPathShortCircuits:
    """An already-imported lerobot is answered without a lookup at all."""

    def test_the_spec_lookup_is_not_reached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
        _patch_find_spec(monkeypatch, _raises(AssertionError("the fast path must not consult the lookup")))

        assert dr._lerobot_installed() is True


class TestAnAbsentLerobotIsReportedAbsent:
    """A lookup that finds nothing is the only way the probe answers False."""

    def test_a_lookup_that_finds_nothing_answers_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_spec_lookup(monkeypatch)
        _patch_find_spec(monkeypatch, lambda _name: None)

        assert dr._lerobot_installed() is False


class TestARaisingLookupIsAnsweredNotPropagated:
    """The probe is called from an error path, so it may not raise on one."""

    @pytest.mark.parametrize(
        "exc",
        [ImportError("a partially-installed namespace package"), ValueError("__spec__ is None")],
        ids=["ImportError", "ValueError"],
    )
    def test_the_failure_is_answered_as_absent(self, exc: BaseException, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_spec_lookup(monkeypatch)
        _patch_find_spec(monkeypatch, _raises(exc))

        assert dr._lerobot_installed() is False

    def test_the_value_error_the_swallow_names_is_one_the_real_lookup_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``ValueError`` is in the ``except`` because the real lookup raises it.

        A module object in ``sys.modules`` whose ``__spec__`` is ``None`` -- what
        a hand-built stand-in for an absent dependency leaves behind -- makes the
        real :func:`importlib.util.find_spec` raise, so the second member of the
        swallowed pair is measured rather than speculative.
        """
        placeholder = types.ModuleType("strands_robots_probe_placeholder")
        placeholder.__spec__ = None
        monkeypatch.setitem(sys.modules, placeholder.__name__, placeholder)

        with pytest.raises(ValueError, match="__spec__ is None"):
            importlib.util.find_spec(placeholder.__name__)


class TestTheProbeDecidesWhichRemedyIsPrinted:
    """The consumer half: the probe's answer selects the install instruction.

    Driven through the real probe rather than a stub of it, so what these pin is
    the link between the lookup's outcome and the remedy a caller reads.
    """

    @staticmethod
    def _diagnose(monkeypatch: pytest.MonkeyPatch, outcome: Callable[[str], Any]) -> str:
        _force_spec_lookup(monkeypatch)
        _patch_find_spec(monkeypatch, outcome)
        return dr._describe_lerobot_import_failure(ModuleNotFoundError("No module named 'lerobot'", name="lerobot"))

    def test_an_absent_lerobot_is_told_to_install_lerobot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        text = self._diagnose(monkeypatch, lambda _name: None)

        assert "lerobot is not installed" in text
        assert "strands-robots[lerobot]" in text

    @pytest.mark.skipif(not _lerobot_resolvable(), reason="lerobot is not installed")
    def test_a_resolvable_lerobot_is_not_told_to_install_lerobot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other side of the branch: naming the extra here is the dead end.

        With lerobot resolvable, the failure has to be attributed to what the
        library imports FROM it, not to lerobot's absence.
        """
        text = self._diagnose(monkeypatch, importlib.util.find_spec)

        assert "lerobot is not installed" not in text
        assert dr._LEROBOT_DATASET_MODULE in text

    def test_a_lookup_failure_does_not_escape_the_diagnosis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A raising lookup must still produce a message, not a second failure.

        The diagnosis is what a caller gets INSTEAD of a traceback, so a probe
        that propagated would replace the actionable text with an unrelated
        exception raised while composing it.
        """
        text = self._diagnose(monkeypatch, _raises(ValueError("lerobot.__spec__ is None")))

        assert "lerobot is not installed" in text
