"""Tests for the in-process execution helpers (no subprocess / shell).

These lock in the security-relevant behavior that replaced the old
``subprocess.run(argv)`` / ``torchrun`` path: upstream code is imported and
CALLED directly (a Python callable - the upstream's own train/run function),
output is captured to a log, and unsafe flag keys are rejected.
"""


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
