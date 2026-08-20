"""A fake's signature is a CLAIM about the real class — this pins the claim (Q56 follow-up).

Q56 was not found by a test. It was found by reading production code, because the test that
covered the broken branch used a fake declaring ``push_to_hub(self, repo_id=None)`` — a parameter
``DatasetRecorder`` has never had. The dashboard called it that way, the fake accepted it, the
suite went green, and the feature ("upload to the Hugging Face Hub after finishing") had never
published anything in its life.

The record path injects its recorder (``recorder_factory``), so EVERY dashboard record test runs
against those fakes and nothing else. That makes their fidelity to the real class load-bearing.

The rule enforced here is deliberately one-directional:

* a fake may accept FEWER parameters than the real method (optional knobs the dashboard never
  passes are not worth mirroring);
* a fake may NEVER accept a parameter name the real method does not have. That is the exact shape
  of the lie — it teaches the suite that a call the real object would reject is fine.

An audit of every ``Fake*``/``Stub*`` class in the tree found this class of divergence only among
recorder-shaped and transport-shaped fakes; the transport ones are stand-ins for MQTT clients,
websockets and ROS publishers (different classes that merely share a method name with Mesh), so
this test scopes itself to the fakes that genuinely stand in for DatasetRecorder.
"""

from __future__ import annotations

import inspect

import pytest

from strands_robots.dataset_recorder import DatasetRecorder
from tests.test_dashboard_record_api import FakeRecorder as ApiFake
from tests.test_dashboard_record_worker import FakeRecorder as WorkerFake


def _params(fn) -> tuple[set[str], bool]:
    sig = inspect.signature(fn)
    names = {n for n in sig.parameters if n != "self"}
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    return names, var_kw


@pytest.mark.parametrize("fake", [ApiFake, WorkerFake], ids=["record_api", "record_worker"])
def test_no_recorder_fake_invents_a_parameter_the_real_recorder_lacks(fake):
    invented: dict[str, set[str]] = {}
    for name, fn in inspect.getmembers(fake, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        real = getattr(DatasetRecorder, name, None)
        if real is None or not inspect.isfunction(real):
            continue
        real_names, real_var_kw = _params(real)
        if real_var_kw:
            continue  # **kwargs accepts anything; a fake cannot lie about it
        fake_names, _ = _params(fn)
        extra = fake_names - real_names
        if extra:
            invented[name] = extra
    assert not invented, (
        f"{fake.__module__}.{fake.__qualname__} accepts parameters DatasetRecorder does not: "
        f"{invented}. A test using this fake would pass while the real recorder raised TypeError — "
        "that is how Q56 (the Hub upload that never worked) survived."
    )


def test_the_audit_itself_can_fail():
    """A guard that cannot fail is decoration. This proves the check has teeth."""

    class LyingFake:
        def push_to_hub(self, repo_id=None):  # the exact Q56 shape
            ...

    with pytest.raises(AssertionError) as exc:
        test_no_recorder_fake_invents_a_parameter_the_real_recorder_lacks(LyingFake)
    assert "repo_id" in str(exc.value)
