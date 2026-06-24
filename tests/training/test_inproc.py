"""Tests for the in-process execution helpers (no subprocess / shell).

These lock in the security-relevant behavior that replaced the old
``subprocess.run(argv)`` / ``torchrun`` path: upstream code is imported and
CALLED (a callable or a module's ``main()``), argv is a LIST, the process
context (sys.argv / cwd / env) is restored afterwards, and unsafe flag keys
are rejected.
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


class TestCallCallable:
    def test_calls_fn_with_args_and_returns(self):
        seen = {}

        def fn(cfg, *, kw=None):
            seen["cfg"] = cfg
            seen["kw"] = kw
            return "done"

        out = _inproc.call_callable(fn, {"a": 1}, kw=7)
        assert out == "done"
        assert seen == {"cfg": {"a": 1}, "kw": 7}

    def test_output_captured_to_log(self, tmp_path):
        log = tmp_path / "run.log"

        def fn():
            print("hello-from-callable")

        _inproc.call_callable(fn, log_path=str(log))
        assert "hello-from-callable" in log.read_text()

    def test_exception_propagates(self):
        def boom():
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            _inproc.call_callable(boom)


class TestCallModuleMain:
    def _write_module(self, tmp_path, name, body):
        pkg = tmp_path / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text(body)
        return f"{name}.mod"

    def test_imports_and_calls_main_with_controlled_argv(self, tmp_path):
        mod = self._write_module(
            tmp_path, "probe_pkg",
            "import sys, json, os\n"
            "def main():\n"
            "    open(os.environ['PROBE_OUT'], 'w').write(json.dumps(sys.argv))\n"
        )
        out = tmp_path / "argv.json"
        sys.path.insert(0, str(tmp_path))
        try:
            _inproc.call_module_main(
                mod, ["--foo=bar", "baz"], env={"PROBE_OUT": str(out)},
            )
        finally:
            sys.path.remove(str(tmp_path))

        import json
        argv = json.loads(out.read_text())
        # argv[0] is the module name; the rest is exactly our LIST (no shell split).
        assert argv[0] == mod
        assert argv[1:] == ["--foo=bar", "baz"]

    def test_process_context_is_restored(self, tmp_path):
        mod = self._write_module(
            tmp_path, "noop_pkg", "def main():\n    pass\n"
        )
        before_argv = list(sys.argv)
        before_cwd = os.getcwd()
        sys.path.insert(0, str(tmp_path))
        try:
            _inproc.call_module_main(mod, ["--x=1"], cwd=str(tmp_path))
        finally:
            sys.path.remove(str(tmp_path))
        assert sys.argv == before_argv
        assert os.getcwd() == before_cwd

    def test_missing_main_raises_attribute_error(self, tmp_path):
        mod = self._write_module(
            tmp_path, "nomain_pkg", "x = 1\n"  # no main()
        )
        sys.path.insert(0, str(tmp_path))
        try:
            with pytest.raises(AttributeError, match="not callable"):
                _inproc.call_module_main(mod, [])
        finally:
            sys.path.remove(str(tmp_path))

    def test_custom_main_attr(self, tmp_path):
        mod = self._write_module(
            tmp_path, "altmain_pkg",
            "called = {}\n"
            "def run_it():\n"
            "    called['yes'] = True\n"
        )
        sys.path.insert(0, str(tmp_path))
        try:
            _inproc.call_module_main(mod, [], main_attr="run_it")
            import importlib
            m = importlib.import_module(mod)
            assert m.called.get("yes") is True
        finally:
            sys.path.remove(str(tmp_path))
