"""Policy(spec) picks a provider from the spec's FORM; ChunkedPolicy replays a chunk."""

import numpy as np
import pytest

from strands_robots import Policy, Robot
from strands_robots.policy import CallablePolicy
from strands_robots.policy.chunked import ChunkedPolicy


def test_a_callable_is_a_policy_and_a_subclass_is_a_plain_constructor():
    assert isinstance(Policy(lambda obs: obs.joints.positions), CallablePolicy)
    assert type(CallablePolicy(lambda obs: 0)) is CallablePolicy


@pytest.mark.parametrize("spec", ["holosoma?", "not a spec", "zenoh://host-without-port/x", "zenoh://h:7447/"])
def test_an_unknown_form_is_refused_listing_the_forms(spec):
    with pytest.raises(ValueError, match=r"zenoh://"):
        Policy(spec)


def test_zenoh_url_and_hub_repo_need_no_network_to_construct():
    pytest.importorskip("lerobot")
    p = Policy("zenoh://10.0.0.5:7447/manipulation", chunk_merge="aligned")
    assert repr(p) == "Policy('zenoh://10.0.0.5:7447/manipulation')"
    assert p.endpoint == "tcp/10.0.0.5:7447" and p.deployment == "manipulation"
    hub = Policy("lerobot/smolvla_base", device="cpu")
    assert isinstance(hub, ChunkedPolicy) and repr(hub.inner) == "Policy('lerobot/smolvla_base')"


class _Count:
    def __init__(self):
        self.calls = 0

    def act_chunk(self, obs, *, instruction=None):
        self.calls += 1
        return np.tile(np.arange(4.0)[:, None], (1, 6)) * 0.01 * self.calls

    def reset(self):
        pass


def test_chunked_policy_plays_the_chunk_then_asks_again():
    inner = _Count()
    arm = Robot("so101")
    arm.run(ChunkedPolicy(inner), steps=8)
    assert inner.calls == 2
    inner2 = _Count()
    arm.run(ChunkedPolicy(inner2, replan_every=2), steps=8)
    assert inner2.calls == 4
