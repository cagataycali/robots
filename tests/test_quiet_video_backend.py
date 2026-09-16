"""torchcodec that cannot load costs one warning line, not a 150-line wall.

LeRobot's default ``video_backend`` resolver imports torchcodec and, when the
wheel is present but its native library cannot load, logs the WHOLE loader
exception (five FFmpeg-version tracebacks) before falling back to pyav.
Measured under ``python -`` (a host the dyld shim refuses to re-exec) on the
first ``stop_recording``: ~150 lines between ``run_policy``'s success and
"Episode saved", for a recording that then read back fine.

``strands_robots._dyld.quiet_video_backend`` asks the same question with
stderr captured and answers once per process: ``"pyav"`` plus one warning
naming the remedy, or ``None`` ("let LeRobot choose") when torchcodec loads or
is not installed. The recorder's create/resume paths and the read-back
constructors pass that answer when the caller named no backend; a backend the
caller passed always wins.
"""

from __future__ import annotations

import importlib
import logging

import pytest

from strands_robots import _dyld


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    monkeypatch.setattr(_dyld, "_quiet_backend", None)
    monkeypatch.setattr(_dyld, "_quiet_backend_probed", False)
    monkeypatch.setattr(_dyld, "_pending_hint", None)


def _fail_import(exc: BaseException):
    real = importlib.import_module

    def fake(name, package=None):
        if name == "torchcodec":
            raise exc
        return real(name, package)

    return fake


class TestTheProbe:
    def test_not_installed_lets_lerobot_choose(self, monkeypatch, caplog):
        monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: False)
        with caplog.at_level(logging.WARNING):
            assert _dyld.quiet_video_backend() is None
        assert caplog.records == []

    def test_a_loading_torchcodec_lets_lerobot_choose(self, monkeypatch, caplog):
        monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
        monkeypatch.setattr(importlib, "import_module", lambda name, package=None: object())
        with caplog.at_level(logging.WARNING):
            assert _dyld.quiet_video_backend() is None
        assert caplog.records == []

    def test_a_torchcodec_that_cannot_load_is_one_line_and_pyav(self, monkeypatch, caplog):
        monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
        wall = "Could not load libtorchcodec. Likely causes:\n" + "\n".join(
            f"FFmpeg version {v}:\nOSError: dlopen(...libtorchcodec_core{v}.dylib): Library not loaded: @rpath/libavutil.dylib"
            for v in (8, 7, 6, 5, 4)
        )
        monkeypatch.setattr(importlib, "import_module", _fail_import(RuntimeError(wall)))
        with caplog.at_level(logging.WARNING, logger="strands_robots._dyld"):
            assert _dyld.quiet_video_backend() == "pyav"
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert msg.startswith(
            "torchcodec is installed but cannot load in this process; decoding video with pyav instead"
        )
        assert "brew install ffmpeg" in msg
        assert "Likely causes" not in msg and "dlopen" not in msg

    def test_the_dyld_hint_is_the_remedy_when_the_shim_has_one(self, monkeypatch, caplog):
        monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
        monkeypatch.setattr(_dyld, "_pending_hint", "export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib")
        monkeypatch.setattr(importlib, "import_module", _fail_import(OSError("Library not loaded: libavutil.59.dylib")))
        with caplog.at_level(logging.WARNING, logger="strands_robots._dyld"):
            assert _dyld.quiet_video_backend() == "pyav"
        assert "export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib" in caplog.records[0].getMessage()

    def test_an_abi_mismatch_names_the_torchcodec_table(self, monkeypatch, caplog):
        monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
        monkeypatch.setattr(importlib, "import_module", _fail_import(ImportError("undefined symbol: _ZN3c10")))
        with caplog.at_level(logging.WARNING, logger="strands_robots._dyld"):
            assert _dyld.quiet_video_backend() == "pyav"
        assert "installing-torchcodec" in caplog.records[0].getMessage()

    def test_a_torchcodec_whose_private_internals_moved_is_not_downgraded(self, monkeypatch, caplog):
        """The probe must import what LeRobot imports: plain ``torchcodec``.

        Naming a private submodule (``torchcodec._core.ops``, where the dlopen
        actually happens) looks more precise and is strictly worse: torchcodec
        is 0.x and that path is private, so the day it moves, the probe raises
        ``ModuleNotFoundError`` - a subclass of ``ImportError``, so it is
        caught - and answers "pyav" with the "built for a different torch"
        remedy for a torchcodec that loads perfectly and that LeRobot's own
        resolver would have chosen. Measured on Thor against a package whose
        ``_core.ops`` was renamed: probing ``torchcodec`` -> None (agreeing with
        LeRobot's "torchcodec"), probing ``torchcodec._core.ops`` -> "pyav".

        Importing ``torchcodec`` loses no coverage: its ``__init__`` imports
        ``._core``, which imports ``.ops``, so the native load is exercised
        either way (asserted below on the real package when it is installed).
        """
        monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
        real = importlib.import_module

        def fake(name, package=None):
            if name == "torchcodec":
                return object()  # the package imports fine
            if name.startswith("torchcodec."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return real(name, package)

        monkeypatch.setattr(importlib, "import_module", fake)
        with caplog.at_level(logging.WARNING, logger="strands_robots._dyld"):
            assert _dyld.quiet_video_backend() is None
        assert caplog.records == []

    def test_importing_torchcodec_covers_the_native_load(self):
        """The reason the top-level name is sufficient, pinned on the real wheel.

        ``torchcodec._core.ops`` is what dlopen()s libtorchcodec; if importing
        the package ever stopped pulling it in, the probe would answer for a
        library it never tried to load.
        """
        pytest.importorskip("torchcodec")
        import sys

        importlib.import_module("torchcodec")
        assert "torchcodec._core.ops" in sys.modules

    def test_probed_once_per_process(self, monkeypatch, caplog):
        monkeypatch.setattr(_dyld, "_torchcodec_installed", lambda: True)
        calls = []

        def counting(name, package=None):
            calls.append(name)
            raise OSError("Library not loaded")

        monkeypatch.setattr(importlib, "import_module", counting)
        with caplog.at_level(logging.WARNING, logger="strands_robots._dyld"):
            assert _dyld.quiet_video_backend() == "pyav"
            assert _dyld.quiet_video_backend() == "pyav"
        assert calls == ["torchcodec"]
        assert len(caplog.records) == 1


def _patch_lerobot_dataset(monkeypatch, fake_cls) -> None:
    import sys
    import types

    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = fake_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)


class _FakeDataset:
    """A LeRobotDataset whose create()/resume() accept ``video_backend``."""

    last_create_kwargs: dict = {}
    last_resume_kwargs: dict = {}

    def __init__(self, repo_id, root=None):
        self.repo_id = repo_id
        self.root = root
        self.meta = type("M", (), {"total_episodes": 0, "total_frames": 0, "fps": 30})()

    @classmethod
    def create(cls, repo_id, fps=30, root=None, robot_type="unknown", features=None, use_videos=True, **kw):
        cls.last_create_kwargs = {"repo_id": repo_id, **kw}
        return cls(repo_id, root=root)

    @classmethod
    def resume(cls, repo_id, root=None, video_backend="auto", **kw):
        cls.last_resume_kwargs = {"repo_id": repo_id, "video_backend": video_backend, **kw}
        return cls(repo_id, root=root)


def _accepting(monkeypatch):
    """Make the fake's create() advertise ``video_backend`` in its signature."""

    def create(
        cls,
        repo_id,
        fps=30,
        root=None,
        robot_type="unknown",
        features=None,
        use_videos=True,
        image_writer_threads=4,
        streaming_encoding=True,
        video_backend="auto",
    ):
        cls.last_create_kwargs = {"repo_id": repo_id, "video_backend": video_backend}
        return cls(repo_id, root=root)

    monkeypatch.setattr(_FakeDataset, "create", classmethod(create))


class TestTheRecorderPassesTheAnswer:
    def test_create_sends_pyav_when_the_caller_named_none(self, monkeypatch, tmp_path):
        from strands_robots import dataset_recorder as dr

        monkeypatch.setattr(dr, "quiet_video_backend", lambda: "pyav")
        _accepting(monkeypatch)
        _patch_lerobot_dataset(monkeypatch, _FakeDataset)
        dr.DatasetRecorder.create("user/data", fps=50, root=str(tmp_path / "ds"), joint_names=["j1"], task="t")
        assert _FakeDataset.last_create_kwargs["video_backend"] == "pyav"

    def test_create_sends_nothing_when_torchcodec_is_fine(self, monkeypatch, tmp_path):
        from strands_robots import dataset_recorder as dr

        monkeypatch.setattr(dr, "quiet_video_backend", lambda: None)
        _accepting(monkeypatch)
        _patch_lerobot_dataset(monkeypatch, _FakeDataset)
        dr.DatasetRecorder.create("user/data", fps=50, root=str(tmp_path / "ds"), joint_names=["j1"], task="t")
        assert _FakeDataset.last_create_kwargs["video_backend"] == "auto"  # the fake's own default: not sent

    def test_a_passed_backend_wins_over_the_probe(self, monkeypatch, tmp_path):
        from strands_robots import dataset_recorder as dr

        monkeypatch.setattr(dr, "quiet_video_backend", lambda: "pyav")
        _accepting(monkeypatch)
        _patch_lerobot_dataset(monkeypatch, _FakeDataset)
        dr.DatasetRecorder.create(
            "user/data", fps=50, root=str(tmp_path / "ds"), joint_names=["j1"], task="t", video_backend="torchcodec"
        )
        assert _FakeDataset.last_create_kwargs["video_backend"] == "torchcodec"

    def test_resume_sends_pyav_when_the_caller_named_none(self, monkeypatch, tmp_path):
        from strands_robots import dataset_recorder as dr

        monkeypatch.setattr(dr, "quiet_video_backend", lambda: "pyav")
        _patch_lerobot_dataset(monkeypatch, _FakeDataset)
        dr.DatasetRecorder.resume("user/data", root=str(tmp_path / "ds"), task="t")
        assert _FakeDataset.last_resume_kwargs["video_backend"] == "pyav"

    def test_read_back_kwargs_follow_the_signature(self, monkeypatch):
        from strands_robots import dataset_recorder as dr

        monkeypatch.setattr(dr, "quiet_video_backend", lambda: "pyav")

        class _NoBackend:
            def __init__(self, repo_id, root=None):
                pass

        class _WithBackend:
            def __init__(self, repo_id, root=None, video_backend=None):
                pass

        assert dr._quiet_backend_kwargs(_NoBackend) == {}
        assert dr._quiet_backend_kwargs(_WithBackend) == {"video_backend": "pyav"}
        monkeypatch.setattr(dr, "quiet_video_backend", lambda: None)
        assert dr._quiet_backend_kwargs(_WithBackend) == {}
