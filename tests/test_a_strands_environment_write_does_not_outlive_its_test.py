"""A ``STRANDS_*`` variable a test writes straight into ``os.environ`` is gone for the next test.

``apply_mesh_env`` writes the environment directly, so a dashboard test that
started a bridge left ``STRANDS_MESH_CAMERA_HZ`` behind for the rest of its
xdist worker and the mesh roster test read a camera loop it never asked for
(#4200). The session fixture in ``tests/conftest.py`` undoes such writes; this
proves it with a nested session where the order is fixed.
"""

from __future__ import annotations

import os

import pytest

pytest_plugins = ("pytester",)

_LEAKER_THEN_VICTIM = """
import os


def test_a_writes_the_environment_directly():
    os.environ["STRANDS_MESH_CAMERA_HZ"] = "20"
    os.environ["STRANDS_MESH_LOCAL_DEV"] = "changed"
    assert os.environ["STRANDS_MESH_CAMERA_HZ"] == "20"


def test_b_reads_the_environment_as_it_was():
    assert "STRANDS_MESH_CAMERA_HZ" not in os.environ
    assert os.environ["STRANDS_MESH_LOCAL_DEV"] == "as-found"
"""


def test_the_conftest_restores_every_strands_variable(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "as-found")
    monkeypatch.delenv("STRANDS_MESH_CAMERA_HZ", raising=False)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conftest = open(os.path.join(root, "tests", "conftest.py"), encoding="utf-8").read()
    start = conftest.index("@pytest.fixture(autouse=True)\ndef _strands_environment_is_left_as_found")
    end = conftest.index("@pytest.fixture(autouse=True)", start + 1)
    fixture_source = conftest[start:end]
    pytester.makeconftest("from collections.abc import Iterator\nimport os\nimport pytest\n\n" + fixture_source)
    pytester.makepyfile(_LEAKER_THEN_VICTIM)
    result = pytester.runpytest("-p", "no:randomly", "-p", "no:cacheprovider", "--no-cov", "-q")
    result.assert_outcomes(passed=2)


def test_the_fixture_itself_undid_a_direct_write_in_this_session() -> None:
    # Half of the proof lives in the parent session: write directly here ...
    os.environ["STRANDS_TEST_LEAK_PROBE"] = "1"
    assert os.environ["STRANDS_TEST_LEAK_PROBE"] == "1"


def test_the_write_from_the_previous_test_is_not_visible_when_order_holds() -> None:
    # ... and, when this runs after the probe (file order; random order may put it
    # first, which is also a pass), the probe is gone. The nested session above
    # is the order-independent proof.
    assert "STRANDS_TEST_LEAK_PROBE" not in os.environ
