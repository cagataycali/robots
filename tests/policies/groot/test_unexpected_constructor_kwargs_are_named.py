"""A constructor kwarg ``Gr00tPolicy`` does not read is named, not dropped.

``create_policy`` forwards one shared kwargs bag to every provider, so the
constructor tolerates keys it does not own - the contract ``lerobot_local`` and
``remote`` already grades with a WARNING naming the keys. ``Gr00tPolicy``
had the same ``**kwargs`` sink and read it nowhere, so ``denoising_steps=8`` (a
parameter this class once had) or ``strict_key=True`` built a service-mode
client on the defaults with no line anywhere saying the request was never read.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[groot-service]'")
pytest.importorskip("zmq", reason="zmq not installed - pip install 'strands-robots[groot-service]'")

from strands_robots.policies.groot.policy import Gr00tPolicy  # noqa: E402


def test_unexpected_constructor_kwargs_warn_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        policy = Gr00tPolicy(port=19999, denoising_steps=8)
    assert isinstance(policy, Gr00tPolicy)
    assert policy._mode == "service"
    named = [r for r in caplog.records if "ignoring unexpected constructor kwarg" in r.message]
    assert len(named) == 1, caplog.text
    # Grade the key off the record's own argument rather than searching the
    # rendered text: prose that happens to contain the probe key satisfies a
    # caplog.text search whatever the key list rendered as.
    args = named[0].args
    assert isinstance(args, tuple), args
    assert args[0] == ["denoising_steps"], caplog.text


def test_known_kwargs_raise_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        Gr00tPolicy(port=19999, strict_keys=False, timeout_ms=500)
    assert not any("ignoring unexpected constructor kwarg" in r.message for r in caplog.records)
