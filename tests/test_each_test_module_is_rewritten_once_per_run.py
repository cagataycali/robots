"""A distributed run rewrites each test module's asserts once, before its workers start.

:mod:`tests.assertion_rewrite_warmup` fills pytest's own assertion-rewrite cache
from the controller. Two facts make that worth having and safe: the workers find
the cache already written when they import the modules, and what is cached is
exactly what pytest's import hook would have written itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from _pytest.assertion import rewrite

from tests.assertion_rewrite_warmup import modules_to_rewrite, rewrite_into_cache

pytest_plugins = ("pytester",)

_REPO_ROOT = Path(__file__).resolve().parent.parent

_RECORDS_WHEN_THE_WORKER_STARTED = """
import os
import time

from tests.assertion_rewrite_warmup import register_rewrite_warmup


def pytest_configure(config):
    register_rewrite_warmup(config)
    if hasattr(config, "workerinput"):
        os.environ["WORKER_STARTED_AT"] = repr(time.time())
"""

_CHECKS_ITS_OWN_CACHE = """
import os
from pathlib import Path

from _pytest.assertion.rewrite import PYC_TAIL

_PYC = Path(__file__).parent / "__pycache__" / (Path(__file__).name[:-3] + PYC_TAIL)
_WRITTEN_BEFORE_THIS_WORKER_STARTED = _PYC.stat().st_mtime < float(os.environ["WORKER_STARTED_AT"])


def test_the_worker_imported_a_cache_written_before_it_started():
    assert _WRITTEN_BEFORE_THIS_WORKER_STARTED
"""


def test_the_workers_import_a_cache_the_controller_wrote(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    """Without the warm-up each worker rewrites the module itself, after it starts."""
    monkeypatch.setenv("PYTHONPATH", str(_REPO_ROOT))
    pytester.makeconftest(_RECORDS_WHEN_THE_WORKER_STARTED)
    pytester.makepyfile(test_a=_CHECKS_ITS_OWN_CACHE, test_b=_CHECKS_ITS_OWN_CACHE)

    result = pytester.runpytest_subprocess("-p", "xdist", "-n", "2", "-p", "no:cacheprovider", "-p", "no:randomly")

    result.assert_outcomes(passed=2)


def test_the_cache_holds_what_pytests_own_hook_would_write(tmp_path: Path, pytestconfig: pytest.Config):
    """Same code object, read back through pytest's own staleness check."""
    module = tmp_path / "test_sample.py"
    module.write_text("def test_it():\n    value = 1\n    assert value + 1 == 3, 'off by one'\n", encoding="utf-8")

    assert rewrite_into_cache([str(module)]) == 1
    pyc = rewrite.get_cache_dir(module) / (module.name[:-3] + rewrite.PYC_TAIL)
    cached = rewrite._read_pyc(module, pyc)

    assert cached is not None
    assert cached == rewrite._rewrite_test(module, pytestconfig)[1]
    assert rewrite_into_cache([str(module)]) == 0, "a current cache is rewritten again"


def test_only_the_modules_pytest_rewrites_are_warmed(pytester: pytest.Pytester):
    """Test modules and conftests are rewritten on import; a helper module is not."""
    pytester.makeconftest("")
    pytester.makepyfile(test_a="", helper="")
    config = pytester.parseconfig(str(pytester.path))

    assert [os.path.basename(path) for path in modules_to_rewrite(config)] == ["conftest.py", "test_a.py"]
