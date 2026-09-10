# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``blocked`` leaves both import mappings exactly as it found them.

:func:`tests._blocked_module.blocked` displaces two entries to make an optional
dependency unimportable - ``sys.modules`` and
:data:`strands_robots.utils._lazy_modules` - and a block that restores only one
of them, or that ``del``\\ s a key it should have put back, does not simply
undo less. It orphans every reference already bound to that module, so a
sibling's ``monkeypatch.setattr`` lands on an object the code under test will
never reach and the sibling passes while grading nothing.

Driven on a throwaway module rather than a real optional dependency, so the
mapping arithmetic is graded on every install: what matters is the round trip,
not which package is missing.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from strands_robots import utils
from strands_robots.utils import require_optional
from tests._blocked_module import blocked

_NAME = "strands_blocked_module_probe"


@pytest.fixture
def probe(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """An importable throwaway module, absent from both mappings before and after."""
    (tmp_path / f"{_NAME}.py").write_text("value = 'original'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, _NAME, raising=False)
    monkeypatch.delitem(utils._lazy_modules, _NAME, raising=False)
    sys.modules.pop(_NAME, None)
    utils._lazy_modules.pop(_NAME, None)
    yield _NAME


def _state(name: str) -> tuple[object, object]:
    """Both mappings' entries for *name*, with a marker for "no entry"."""
    return (sys.modules.get(name, "absent"), utils._lazy_modules.get(name, "absent"))


class TestTheBlockBites:
    """It has to actually make the dependency missing, or it grades nothing."""

    def test_require_optional_refuses_inside_the_block(self, probe: str) -> None:
        require_optional(probe)  # memoise it first: the block must beat the memo

        with blocked(probe), pytest.raises(ImportError) as caught:
            require_optional(probe)

        assert caught.value.name == probe

    def test_a_plain_import_fails_inside_the_block(self, probe: str) -> None:
        with blocked(probe), pytest.raises(ImportError):
            __import__(probe)


def _raise_from_inside() -> None:
    """Raise from inside a ``blocked`` body without ending the test's own flow.

    A literal ``raise`` under ``pytest.raises`` reads as a terminal statement to a
    control-flow reader that does not model the context manager, which makes the
    assertion after it look unreachable. The behaviour under test is the same.
    """
    raise RuntimeError("from inside")


class TestTheRoundTrip:
    """Every starting state comes back unchanged."""

    @pytest.mark.parametrize(
        "prepare",
        [
            pytest.param(lambda name: None, id="absent-from-both"),
            pytest.param(__import__, id="imported-not-memoised"),
            pytest.param(require_optional, id="imported-and-memoised"),
        ],
    )
    def test_both_mappings_are_left_as_they_were_found(self, probe: str, prepare) -> None:
        prepare(probe)
        before = _state(probe)

        with blocked(probe):
            pass

        assert _state(probe) == before

    def test_an_exception_inside_the_block_still_restores(self, probe: str) -> None:
        require_optional(probe)
        before = _state(probe)

        with pytest.raises(RuntimeError, match="from inside"), blocked(probe):
            _raise_from_inside()

        assert _state(probe) == before

    def test_a_module_the_block_found_absent_is_left_absent(self, probe: str) -> None:
        """Not merely un-overwritten: ``None`` must not be left behind as the entry."""
        with blocked(probe):
            pass

        assert probe not in sys.modules
        assert probe not in utils._lazy_modules


def test_a_patched_module_survives_the_block(probe: str) -> None:
    """The regression: a double installed before the block is still the one consulted.

    This is the failure the ``del``-ing copies produced. The patch goes on the
    module object a caller already holds; if the block leaves either mapping
    naming a freshly imported object, ``require_optional`` answers with a module
    that never saw the patch and the assertion it was installed for grades
    nothing.
    """
    held = require_optional(probe)
    held.value = "double"  # type: ignore[attr-defined]

    with blocked(probe):
        pass

    assert require_optional(probe) is held
    assert __import__(probe) is held
    assert require_optional(probe).value == "double"  # type: ignore[attr-defined]
