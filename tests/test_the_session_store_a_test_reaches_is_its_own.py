# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""No test reaches the session records of a run this machine has going.

:data:`strands_robots.tools._process_stop.SESSION_DIR` is
``Path.cwd() / ".strands_robots/.sessions"``, resolved as the module is
imported, so a suite run from a project directory resolves it to that project's
real store. The store is one document: ``SessionManager`` loads every record to
change one and writes them all back, and ``remove_session`` deletes out of it -
so a ``stop`` or ``list`` cell reaching it takes the operator's live
teleoperation and training records with it, under a green report, because
nothing in those cells asks where the store was.

Twenty-four modules redirected it for themselves. This module declares no
fixture of its own on purpose: the guarantee under test is that the redirect is
a property of the session - the ``_the_session_store_a_test_reaches_is_its_own``
fixture in ``tests/conftest.py`` - rather than of twenty-four authors having
remembered to. Every cell below fails on a tree without that fixture.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("psutil")

from strands_robots.tools import _process_stop  # noqa: E402

# The names in ``strands_robots.tools`` are the decorated tools; the modules
# that hold ``SessionManager`` are reached by their dotted path.
tele_mod = importlib.import_module("strands_robots.tools.lerobot_teleoperate")
train_mod = importlib.import_module("strands_robots.tools.lerobot_train")

#: The two session tools, which share the one store on purpose.
TOOLS = [pytest.param(tele_mod, id="teleoperate"), pytest.param(train_mod, id="train")]


def test_the_session_directory_is_under_this_test_s_own_tmp_path(tmp_path: Path) -> None:
    """The name itself, before anything reads or writes through it."""
    session_dir = _process_stop.SESSION_DIR

    assert session_dir.is_relative_to(tmp_path), f"a test resolved the session store to {session_dir}"
    assert not session_dir.is_relative_to(Path.cwd()), "the session store resolved inside the working tree"


@pytest.mark.parametrize("module", TOOLS)
def test_a_manager_a_tool_builds_holds_this_test_s_store(tmp_path: Path, module: ModuleType) -> None:
    """``SessionManager`` binds the path when it is constructed, not per call."""
    store = module.SessionManager().sessions_file

    assert store.is_relative_to(tmp_path), f"a session manager bound {store}"


def test_a_log_path_is_created_under_this_test_s_own_tmp_path(tmp_path: Path) -> None:
    """``session_log_path`` creates the directory it names, wherever that is."""
    log = _process_stop.session_log_path("a-session")

    assert log.is_relative_to(tmp_path), f"a session log was placed at {log}"
    assert log.parent.is_dir()


@pytest.mark.parametrize("module", TOOLS)
def test_a_record_a_test_writes_is_not_visible_to_the_next_test(tmp_path: Path, module: ModuleType) -> None:
    """The redirect is per test, so a record written here cannot outlive it.

    Written through the tool's own manager and read back off disk, so the
    assertion is about the file rather than about an in-process cache.
    """
    module.SessionManager().add_session("a-run", {"pid": os.getpid(), "start_time": 0.0, "action": "teleoperate"})

    store = tmp_path / ".sessions" / "active_sessions.json"
    assert json.loads(store.read_text(encoding="utf-8")).keys() == {"a-run"}
