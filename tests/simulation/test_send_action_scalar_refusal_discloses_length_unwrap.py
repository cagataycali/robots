"""The scalar-shape refusal discloses the length-1 unwrap, so a caller
reading 'got list' for ``[0.5, 0.6]`` can see that ``[0.5]`` would have
worked.

Context (``strands_robots/simulation/base.py``):

* ``_unwrap_single_element_action_value`` (base.py:839) is a deliberate
  contract from #1538 (the GR00T list-form regression): a 1-DOF key emitted
  by ``Policy.get_actions`` as ``[v]`` is unwrapped to its one scalar
  rather than rejected. Multi-element values are still the #1179 crash
  class and still rejected.
* The refusal names a "scalar number (one value per actuator/joint)"
  rule. Before this guard, a caller who hit the rejection on
  ``[0.5, 0.6]`` was told only the type - not that a length-1 sequence
  would have been accepted. That is the half-rule the message closes.

The hint is scoped to ``sequence_length(value) > 1`` for a non-mapping,
non-str/bytes value, because:

* a 0-d numpy array has no length ``sequence_length`` can read (returns
  ``None``), and already is a scalar for the unwrap;
* a mapping is a shape-level mistake (``dict``) that would not be helped
  by the length-1 remark;
* a str/bytes carries characters that would mis-read as 'elements'.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


@pytest.fixture()
def so101_sim():
    sim = MuJoCoSimEngine(tool_name="so101_sim")
    sim.create_world()
    sim.add_robot(name="so101", data_config="so101")
    try:
        yield sim
    finally:
        sim.destroy()


class TestLengthOneUnwrapIsDisclosed:
    """The refusal names the unwrap when it would have helped."""

    def test_multi_element_list_names_length(self, so101_sim):
        res = so101_sim.send_action({"1": [0.5, 0.6]})
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        assert "scalar number" in text
        assert "got list" in text
        # The new disclosure: the actual length AND the length-1 escape hatch.
        assert "carries 2 elements" in text
        assert "length-1 sequence" in text

    def test_multi_element_tuple_names_length(self, so101_sim):
        res = so101_sim.send_action({"1": (0.5, 0.6, 0.7)})
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        assert "carries 3 elements" in text
        assert "length-1 sequence" in text

    def test_multi_element_ndarray_names_length(self, so101_sim):
        res = so101_sim.send_action({"1": np.array([0.5, 0.6, 0.7, 0.8])})
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        assert "carries 4 elements" in text
        assert "length-1 sequence" in text


class TestLengthOneUnwrapHintIsScopedCorrectly:
    """The disclosure applies only where it would help."""

    def test_dict_does_not_claim_length(self, so101_sim):
        res = so101_sim.send_action({"1": {"v": 0.5}})
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        # Dict is a shape mistake, not an unwrap miss; no "N elements" claim.
        assert "scalar number" in text
        assert "got dict" in text
        assert "elements" not in text

    def test_unparseable_string_does_not_claim_length(self, so101_sim):
        # 'hello' fails ``float()`` and ``sequence_length("hello")`` is None
        # (``str``/``bytes`` are explicitly handled upstream, so a value that
        # reaches here is already not iterated as a sequence).
        res = so101_sim.send_action({"1": "hello"})
        assert res["status"] == "error"
        text = res["content"][0]["text"]
        assert "scalar number" in text
        assert "got str" in text
        # No stray length chatter for a value that is not iterable-as-vector.
        assert "elements" not in text


class TestLengthOneIsStillAccepted:
    """The unwrap contract (#1538) continues to hold."""

    @pytest.mark.parametrize(
        "value",
        [[0.5], (0.5,), np.array([0.5])],
    )
    def test_length_one_sequence_accepted(self, so101_sim, value):
        res = so101_sim.send_action({"1": value})
        assert res["status"] == "success", res
