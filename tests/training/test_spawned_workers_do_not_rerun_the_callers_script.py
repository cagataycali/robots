"""A process spawned while training runs in-process does not re-run the caller's script.

LeRobot starts its DataLoader workers with ``spawn``, and a spawn child
re-imports the parent's ``__main__``. Training runs in-process through
``call_callable`` / ``elastic_launch_callable``, so an unguarded caller script,
which is the shape of every agent script, ran again once per worker: a marker
line at its top ran 5 times for one ``train_policy`` call. The cells run a real
unguarded script in a fresh interpreter and count how often its body ran.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from strands_robots.training import _inproc

_SCRIPT = textwrap.dedent(
    """
    import multiprocessing as mp
    import os

    with open(os.environ["MARKER"], "a") as f:
        f.write("ran\\n")

    from strands_robots.training._inproc import call_callable


    def train():
        ctx = mp.get_context("spawn")
        workers = [ctx.Process(target=os.getpid) for _ in range(2)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        return [w.exitcode for w in workers]


    print("EXIT", call_callable(train))
    """
)


def _run(tmp_path: Path, **env: str) -> tuple[int, str]:
    script = tmp_path / "agent_script.py"
    script.write_text(_SCRIPT)
    marker = tmp_path / "marker.txt"
    repo = Path(__file__).resolve().parents[2]
    full_env = {k: v for k, v in os.environ.items() if k != "STRANDS_TRAIN_WORKERS_IMPORT_MAIN"}
    full_env.update(MARKER=str(marker), PYTHONPATH=str(repo), **env)
    out = subprocess.run([sys.executable, str(script)], env=full_env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return marker.read_text().count("ran"), out.stdout


def test_an_unguarded_script_runs_once_while_workers_are_spawned(tmp_path):
    runs, stdout = _run(tmp_path)
    assert runs == 1
    assert "EXIT [0, 0]" in stdout  # the workers themselves ran and exited cleanly


def test_the_opt_out_keeps_pythons_default(tmp_path):
    runs, _ = _run(tmp_path, STRANDS_TRAIN_WORKERS_IMPORT_MAIN="1")
    assert runs == 3  # the parent and each of the two workers


def test_main_is_restored_after_the_call_and_after_a_failure(monkeypatch):
    main = types.ModuleType("__main__")
    main.__file__ = "/tmp/agent_script.py"
    main.__spec__ = types.SimpleNamespace(name="agent_script")
    monkeypatch.setitem(sys.modules, "__main__", main)
    monkeypatch.delenv("STRANDS_TRAIN_WORKERS_IMPORT_MAIN", raising=False)
    seen = []

    def peek():
        seen.append((main.__dict__.get("__spec__"), main.__dict__.get("__file__")))

    _inproc.call_callable(peek)
    assert seen == [(None, None)]
    assert main.__file__ == "/tmp/agent_script.py" and main.__spec__.name == "agent_script"

    def boom():
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError):
        _inproc.call_callable(boom)
    assert main.__file__ == "/tmp/agent_script.py" and main.__spec__.name == "agent_script"


def test_an_interactive_main_without_a_file_is_left_as_it_was(monkeypatch):
    main = types.ModuleType("__main__")
    main.__spec__ = None
    monkeypatch.setitem(sys.modules, "__main__", main)
    _inproc.call_callable(lambda: None)
    assert "__file__" not in main.__dict__ and main.__spec__ is None
