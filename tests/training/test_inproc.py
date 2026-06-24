"""Tests for the in-process execution helpers (no subprocess / shell).

These lock in the security-relevant behavior that replaced the old
``subprocess.run(argv)`` / ``torchrun`` path: argv is a LIST, runpy runs the
target IN this interpreter with a restored process context, and unsafe flag
keys are rejected.
"""

import os
import sys

import pytest

from strands_robots.training import _inproc


class TestSafeFlagKey:
    @pytest.mark.parametrize("key", [
        "weight_decay", "lr", "trainer.max_iter",
        "dataloader_train.dataloader.datasets.droid.use_filter_dict",
        "a-b_c.d", "x0",
    ])
    def test_accepts_safe_keys(self, key):
        assert _inproc.safe_flag_key(key) is True

    @pytest.mark.parametrize("key", [
        "", "-leading", "--inject", "has space", "rm;rf",
        "$(boom)", "a|b", "a&b", "a\nb", ".dotfirst", "k=v",
    ])
    def test_rejects_unsafe_keys(self, key):
        assert _inproc.safe_flag_key(key) is False

    def test_filter_safe_extra_splits_and_drops_consumed(self):
        extra = {
            "lr": 1e-4,
            "consumed_key": "x",
            "bad key; rm -rf /": "boom",
            "trainer.max_iter": 10,
        }
        safe, rejected = _inproc.filter_safe_extra(extra, consumed={"consumed_key"})
        assert safe == {"lr": 1e-4, "trainer.max_iter": 10}
        assert rejected == ["bad key; rm -rf /"]


class TestRunPythonModule:
    def test_runs_in_process_with_controlled_argv(self, tmp_path):
        # A tiny module on a temp sys.path that records its argv to a file.
        pkg = tmp_path / "inproc_probe"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            "import sys, json, os\n"
            "open(os.environ['PROBE_OUT'], 'w').write(json.dumps(sys.argv))\n"
        )
        out = tmp_path / "argv.json"
        sys.path.insert(0, str(tmp_path))
        try:
            _inproc.run_python_module(
                "inproc_probe",
                ["--foo=bar", "baz"],
                env={"PROBE_OUT": str(out)},
            )
        finally:
            sys.path.remove(str(tmp_path))

        import json
        argv = json.loads(out.read_text())
        # runpy(alter_sys=True) sets argv[0] to the module's __main__.py path;
        # what matters for safety is argv[1:] == exactly our LIST (no shell split,
        # no extra tokens, no interpolation).
        assert argv[0].endswith("__main__.py")
        assert argv[1:] == ["--foo=bar", "baz"]

    def test_process_context_is_restored(self, tmp_path):
        pkg = tmp_path / "noop_probe"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text("pass\n")
        before_argv = list(sys.argv)
        before_cwd = os.getcwd()
        sys.path.insert(0, str(tmp_path))
        try:
            _inproc.run_python_module("noop_probe", ["--x=1"], cwd=str(tmp_path))
        finally:
            sys.path.remove(str(tmp_path))
        # argv / cwd must be exactly as before (no leakage from the run).
        assert sys.argv == before_argv
        assert os.getcwd() == before_cwd

    def test_nonzero_systemexit_propagates(self, tmp_path):
        pkg = tmp_path / "fail_probe"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text("import sys; sys.exit(3)\n")
        sys.path.insert(0, str(tmp_path))
        try:
            with pytest.raises(SystemExit):
                _inproc.run_python_module("fail_probe", [])
        finally:
            sys.path.remove(str(tmp_path))

    def test_clean_exit_is_swallowed(self, tmp_path):
        pkg = tmp_path / "ok_probe"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text("import sys; sys.exit(0)\n")
        sys.path.insert(0, str(tmp_path))
        try:
            _inproc.run_python_module("ok_probe", [])  # must NOT raise
        finally:
            sys.path.remove(str(tmp_path))


class TestRunPythonPath:
    def test_runs_script_file_in_process(self, tmp_path):
        script = tmp_path / "probe.py"
        out = tmp_path / "argv.json"
        script.write_text(
            "import sys, json, os\n"
            "open(os.environ['PROBE_OUT'], 'w').write(json.dumps(sys.argv))\n"
        )
        _inproc.run_python_path(
            str(script), ["--a=1", "--b=2"], env={"PROBE_OUT": str(out)},
        )
        import json
        argv = json.loads(out.read_text())
        assert argv[0] == str(script)
        assert argv[1:] == ["--a=1", "--b=2"]
