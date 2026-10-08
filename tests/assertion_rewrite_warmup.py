"""Rewrite each test module's asserts once per run, not once per worker.

pytest rewrites the ``assert`` statements of every test module it imports and
caches the result in a ``__pycache__`` file. A fresh checkout has no cache, and
every ``pytest-xdist`` worker collects the whole tree in the same order at the
same moment, so each worker rewrites all ~2,200 modules itself: about a minute
of CPU per worker, before the first test starts. Measured on two cores, a cold
collection under ``-n 2`` took 144 s and a warm one 79 s.

The controller spawns no worker until ``pytest_sessionstart`` returns, so this
plugin rewrites the modules there instead - split across as many processes as
the session will have workers - and every worker then reads the cache. The
cache file is pytest's own: it is written by pytest's own rewrite, checked by
pytest's own staleness test, so a module edited after the warm-up is rewritten
again by the worker that imports it, as it always was.

The plugin does nothing outside a distributed controller, when bytecode is not
written (``PYTHONDONTWRITEBYTECODE``), or when ``enable_assertion_pass_hook`` is
on - the rewrite the warm-up runs has no session ``Config`` to read that option
from, so it would cache a module without the hook calls.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import pytest
from _pytest.assertion import rewrite
from _pytest.pathlib import fnmatch_ex

#: Name the plugin is registered under, so a second registration is a no-op.
PLUGIN_NAME = "assertion_rewrite_warmup"

#: What pytest's cache writer reads off its session state: only ``trace``, on a failed write.
_NO_TRACE = SimpleNamespace(trace=lambda message: None)


def rewrite_into_cache(paths: list[str]) -> int:
    """Rewrite each module in *paths* the way pytest's import hook does, and cache it.

    Args:
        paths: Test module files.

    Returns:
        How many modules were rewritten; one whose cache is already current is skipped.
    """
    written = 0
    for name in paths:
        path = Path(name)
        cache_dir = rewrite.get_cache_dir(path)
        if not rewrite.try_makedirs(cache_dir):
            continue
        pyc = cache_dir / (path.name[:-3] + rewrite.PYC_TAIL)
        if rewrite._read_pyc(path, pyc) is not None:
            continue
        try:
            source_stat, code = rewrite._rewrite_test(path, None)  # type: ignore[arg-type]
        except (OSError, SyntaxError, ValueError):
            # The worker that imports it reports the error, as it does without this.
            continue
        if rewrite._write_pyc(_NO_TRACE, code, source_stat, pyc):  # type: ignore[arg-type]
            written += 1
    return written


def modules_to_rewrite(config: pytest.Config) -> list[str]:
    """The test modules and conftests under the session's paths that pytest will rewrite.

    Args:
        config: The session configuration; its ``python_files`` patterns pick the modules.

    Returns:
        Their paths, sorted.
    """
    patterns = config.getini("python_files")
    roots = [Path(str(arg).split("::", 1)[0]) for arg in config.args]
    found: set[str] = set()
    for root in (r if r.is_absolute() else Path(config.invocation_params.dir, r) for r in roots):
        candidates = [root] if root.is_file() else root.rglob("*.py")
        for path in candidates:
            if path.suffix == ".py" and (
                path.name == "conftest.py" or any(fnmatch_ex(pattern, path) for pattern in patterns)
            ):
                found.add(str(path))
    return sorted(found)


class AssertionRewriteWarmup:
    """Fill pytest's assertion-rewrite cache before the workers start."""

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionstart(self, session: pytest.Session) -> None:
        """Rewrite every module the workers will import, across one process per worker."""
        config = session.config
        workers = len(getattr(config.option, "tx", None) or ())
        if (
            workers < 2
            or hasattr(config, "workerinput")
            or sys.dont_write_bytecode
            or config.getini("enable_assertion_pass_hook")
        ):
            return
        paths = modules_to_rewrite(config)
        chunks = [paths[index::workers] for index in range(workers)]
        # spawn, not fork: the controller has already imported the suite's
        # conftest and its threads, and a forked copy of those is not this
        # function's concern.
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            list(pool.map(rewrite_into_cache, chunks))


def register_rewrite_warmup(config: pytest.Config) -> None:
    """Register :class:`AssertionRewriteWarmup` on *config* once."""
    if not config.pluginmanager.has_plugin(PLUGIN_NAME):
        config.pluginmanager.register(AssertionRewriteWarmup(), PLUGIN_NAME)
