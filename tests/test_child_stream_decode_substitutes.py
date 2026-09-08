r"""A child process's stream is decoded with substitution, not strictly.

``subprocess`` in text mode decodes with ``errors="strict"`` unless told
otherwise, so one byte the codec cannot decode replaces the whole captured
stream with ``UnicodeDecodeError``. A child's stdout is not this process's
text: it is whatever bytes an arbitrary program wrote to a pipe - ffmpeg
copying a latin-1 metadata tag, a container printing a log line byte for byte,
a USB descriptor string a vendor chose - so a byte that is not valid UTF-8 is
expected there, not exceptional.

The package had already decided that, twice, for the *same* bytes:

* ``lerobot_train`` / ``lerobot_teleoperate`` read the detached child's log file
  with ``errors="replace"``, and say why - reading it strictly makes one such
  byte raise past the handler that exists to keep the rest of the report.
* the MuJoCo GL probe in ``simulation.mujoco.backend`` captures the child as
  bytes and calls ``.decode(errors="replace")`` on both streams.

The 25 sites that read a child through a *pipe* in text mode did not, and two
of them lose something a caller was promised:

* :func:`~strands_robots.dataset_recorder.sync_dataset_to_bucket` documents
  "Never raises on ``hf`` failure; errors are surfaced in the result dict", and
  ``stop_recording`` calls it unguarded on that promise. The decode happens
  after ``hf sync`` has already run, so an undecodable byte in the CLI's own
  message raised past a completed upload and took the episode/frame report with
  it.
* ``VeraServerRunner``'s log pump iterates the server's stdout on a daemon
  thread. The byte killed that thread, silently, and the readiness error that
  follows tells an operator to "Check the [vera.server] log lines above" - lines
  that stopped arriving at the byte.

Two boundaries are deliberate. ``encoding=`` is *not* stated: a child inherits
this process's locale and encodes with it, which is the one case
tests/test_on_disk_text_io_states_utf8.py excludes for that reason. And a pipe
this process *writes* stays strict - substituting on the way out would silently
mangle a command instead of refusing to send it.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import subprocess
import sys
from typing import Any

import pytest

import strands_robots
from strands_robots import dataset_recorder as recorder_mod
from strands_robots.policies.vera import VeraConfig
from strands_robots.policies.vera import server_runner as sr

PACKAGE = pathlib.Path(strands_robots.__file__).parent

#: The ``subprocess`` entry points that spawn a child.
SPAWNERS = frozenset({"subprocess.run", "subprocess.Popen", "subprocess.check_output", "subprocess.check_call"})

#: What ``errors="replace"`` substitutes for a byte the codec cannot decode.
REPLACEMENT = "\ufffd"

#: A child that writes exactly the bytes a test names, then exits with a chosen
#: code. Byte payloads travel as hex so this stays ASCII source.
_CHILD = (
    "import sys;"
    "sys.stdout.buffer.write(bytes.fromhex(sys.argv[1]));"
    "sys.stderr.buffer.write(bytes.fromhex(sys.argv[2]));"
    "sys.stdout.buffer.flush();"
    "sys.stderr.buffer.flush();"
    "sys.exit(int(sys.argv[3]))"
)

#: The hub's own wording for the re-sync case, with one byte that is not UTF-8.
#: ``sync_dataset_to_bucket`` matches this text to decide a pre-existing bucket
#: is not an error, so the byte lands in a string the code reads, not just one
#: it reports.
UNDECODABLE_CLI_LINE = b"You already created this bucket repo \xff\n"


def _child_argv(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> list[str]:
    """Argv for a child that emits *stdout* / *stderr* verbatim and exits."""
    return [sys.executable, "-c", _CHILD, stdout.hex(), stderr.hex(), str(returncode)]


def _reads_a_child_stream(keywords: dict[str, str]) -> bool:
    """Whether this call reads a child's output rather than only writing to it.

    ``capture_output=True`` or an explicit ``PIPE`` on stdout/stderr gives the
    parent a stream to decode. A call whose only pipe is ``stdin`` writes, and a
    call that redirects to a file has no stream in this process at all.
    """
    if keywords.get("capture_output") == "True":
        return True
    return any(keywords.get(stream) == "subprocess.PIPE" for stream in ("stdout", "stderr"))


def _scan() -> tuple[list[str], list[str]]:
    """(child-stream reads that decode strictly, all child-stream reads found)."""
    strict: list[str] = []
    reads: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or ast.unparse(node.func) not in SPAWNERS:
                continue
            keywords = {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
            text_mode = any(name in keywords for name in ("text", "universal_newlines")) or "encoding" in keywords
            if not text_mode or not _reads_a_child_stream(keywords):
                continue
            site = f"{path.relative_to(PACKAGE.parent)}:{node.lineno}"
            reads.append(site)
            if "errors" not in keywords:
                strict.append(site)
    return strict, reads


class TestEveryChildStreamReadSubstitutes:
    """The rule, graded across the package so a new call site is caught."""

    def test_no_read_of_a_child_stream_decodes_strictly(self) -> None:
        strict, _ = _scan()
        assert not strict, (
            "these calls decode a child's stream with errors='strict', so one byte the "
            "codec cannot decode raises instead of being substituted; pass "
            'errors="replace" as the log-file readers and the GL probe already do:\n  ' + "\n  ".join(strict)
        )

    def test_the_scan_finds_the_reads_it_grades(self) -> None:
        # Without this a matcher that silently stops matching passes the rule
        # above by finding nothing to grade.
        _, reads = _scan()
        assert len(reads) >= 20, f"only {len(reads)} child-stream reads found; the matcher is broken"


class TestBucketSyncKeepsItsVerdict:
    """``sync_dataset_to_bucket`` reports an undecodable CLI message.

    Its docstring promises a result dict on any ``hf`` failure and
    ``stop_recording`` calls it unguarded, after the upload has already run.
    """

    @pytest.fixture
    def wire(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        """Every ``hf`` argv runs as a real child, with the production kwargs.

        The argv is swapped for a controlled child but the keyword arguments are
        handed through untouched, so the decode under test is the shipped one.
        """
        recorded: list[list[str]] = []
        real_run = subprocess.run

        def run(cmd: Any, **kwargs: Any) -> Any:
            recorded.append(list(cmd))
            return real_run(_child_argv(stdout=UNDECODABLE_CLI_LINE), **kwargs)

        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(recorder_mod, "_hf_executable", lambda: "/usr/bin/hf")
        monkeypatch.setattr(recorder_mod, "_huggingface_hub_version_error", lambda: None)
        return recorded

    @pytest.fixture
    def finalized(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """A dataset root that passes the ``meta/`` finalization check."""
        (tmp_path / "meta").mkdir()
        return tmp_path

    @pytest.mark.parametrize("create", [True, False], ids=["bucket-create", "sync-only"])
    def test_an_undecodable_cli_message_is_reported_not_raised(
        self, wire: list[list[str]], finalized: pathlib.Path, create: bool
    ) -> None:
        result = recorder_mod.sync_dataset_to_bucket(finalized, "acme/robotdata", run_id="run1", create=create)
        assert result["status"] == "success", result
        assert result["bucket_uri"] == "hf://buckets/acme/robotdata/run1"
        assert wire, "no hf argv reached the wire"

    def test_a_failing_sync_keeps_the_message_around_the_byte(
        self, monkeypatch: pytest.MonkeyPatch, finalized: pathlib.Path
    ) -> None:
        real_run = subprocess.run

        def run(cmd: Any, **kwargs: Any) -> Any:
            return real_run(_child_argv(stderr=b"quota exceeded for \xffacme/robotdata\n", returncode=1), **kwargs)

        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(recorder_mod, "_hf_executable", lambda: "/usr/bin/hf")
        monkeypatch.setattr(recorder_mod, "_huggingface_hub_version_error", lambda: None)

        result = recorder_mod.sync_dataset_to_bucket(finalized, "acme/robotdata", run_id="run1", create=False)
        assert result["status"] == "error"
        # The report keeps the diagnosis on both sides of the byte.
        assert "quota exceeded for" in result["message"]
        assert "acme/robotdata" in result["message"]
        assert REPLACEMENT in result["message"]

    def test_a_decodable_message_is_unchanged(self, monkeypatch: pytest.MonkeyPatch, finalized: pathlib.Path) -> None:
        # The non-narrowing control: substitution only touches bytes that had no
        # decoding at all.
        real_run = subprocess.run

        def run(cmd: Any, **kwargs: Any) -> Any:
            return real_run(_child_argv(stderr=b"quota exceeded for acme/robotdata\n", returncode=1), **kwargs)

        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(recorder_mod, "_hf_executable", lambda: "/usr/bin/hf")
        monkeypatch.setattr(recorder_mod, "_huggingface_hub_version_error", lambda: None)

        result = recorder_mod.sync_dataset_to_bucket(finalized, "acme/robotdata", run_id="run1", create=False)
        assert result["message"] == "quota exceeded for acme/robotdata"
        assert REPLACEMENT not in result["message"]


class TestTheServerLogPumpOutlivesTheByte:
    """The VERA log pump keeps streaming past a byte it cannot decode.

    The readiness failure tells an operator to read these lines, so a pump that
    dies at the byte takes the diagnosis with it.
    """

    def test_lines_after_an_undecodable_byte_still_reach_the_log(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        server_output = b"loading checkpoint /ckpt/caf\xff.pt\nserver ready\n"
        monkeypatch.setattr(sr, "_require_vera_installed", lambda _executable: None)
        monkeypatch.setattr(sr.VeraServerRunner, "_build_command", lambda self: _child_argv(stdout=server_output))
        monkeypatch.setattr(sr.VeraServerRunner, "_wait_until_ready", lambda self: None)

        runner = sr.VeraServerRunner(VeraConfig(server_port=8820))
        with caplog.at_level(logging.INFO, logger=sr.__name__):
            runner.start()
            assert runner._log_thread is not None
            runner._log_thread.join(timeout=30)
            assert not runner._log_thread.is_alive(), "the log pump did not finish"
        runner.stop()

        pumped = [record.getMessage() for record in caplog.records if "[vera.server]" in record.getMessage()]
        assert any(REPLACEMENT in line for line in pumped), pumped
        # The line after the byte is the one a dead pump loses.
        assert any("server ready" in line for line in pumped), pumped
