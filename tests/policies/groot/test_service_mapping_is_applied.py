"""A GR00T service-mode caller's key mappings are honoured, not discarded.

``Gr00tPolicy`` accepts ``observation_mapping`` and ``action_mapping`` on both
transports, and they are the only way to drive a model whose channel names
differ from the robot's. Resolving them was reached only after a local model
loaded, so on the service transport both were stored and never parsed - and
neither consumer of an unparsed mapping reports anything:

* the flat wire builder looked every declared channel up under the model's own
  name, so a robot naming its camera ``wrist_cam`` for ``video.wrist`` matched
  nothing and the payload carried the instruction alone - a server receiving no
  video and no state at all, under a successful call;
* the action unpacker skips renaming on a falsy mapping, so the caller's
  requested actuator names were absent from the returned steps.

Both halves are silent, which is what these cases pin. The mapped-channel
assertions are the load-bearing ones; the unmapped-caller cases are controls
against the opposite error, resolving *more* than the caller declared.

Nothing here needs the ``groot-service`` extra: the client is replaced with a
recorder before construction, so the real ``__init__`` runs - which is where
mapping resolution is wired - without a ZMQ socket. Routing these through the
extra would skip them exactly where a service-mode user is most likely to be.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.policies.groot import policy as policy_mod
from strands_robots.policies.groot.policy import Gr00tPolicy

#: ``so100`` declares one camera and two state channels, none of which the
#: mapped observations below name - so an ignored mapping resolves nothing and
#: an honoured one resolves exactly what it declares.
_DECLARED_VIDEO = "video.webcam"
_DECLARED_STATE = ("state.single_arm", "state.gripper")
_DECLARED_LANG = "annotation.human.task_description"

_MAPPING_OBS = {"wrist_cam": "video.wrist", "arm_joints": "state.arm"}
_MAPPING_ACTION = {"action.arm": "joints"}


class _RecordingClient:
    """Stands in for ``Gr00tInferenceClient``; records the payload sent."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sent: list[dict] = []

    def get_action(self, observations: dict) -> dict:
        self.sent.append(observations)
        return {"action.arm": np.zeros((1, 2, 3), dtype=np.float32)}


@pytest.fixture
def service_policy(monkeypatch):
    """Build a service-mode policy whose client records instead of dialling."""

    def _build(**kwargs) -> Gr00tPolicy:
        monkeypatch.setattr(policy_mod, "Gr00tInferenceClient", _RecordingClient)
        return Gr00tPolicy(data_config="so100", host="localhost", port=19999, **kwargs)

    return _build


def _mapped_observation() -> dict[str, Any]:
    """Robot-side names from ``_MAPPING_OBS``, none of them model names."""
    return {
        "wrist_cam": np.zeros((8, 8, 3), dtype=np.uint8),
        "arm_joints": [0.1, 0.2, 0.3],
    }


def _identity_observation() -> dict[str, Any]:
    """Robot names each channel exactly as ``so100`` declares it."""
    return {
        "webcam": np.zeros((8, 8, 3), dtype=np.uint8),
        "single_arm": [0.0, 0.1, 0.2, 0.3, 0.4],
        "gripper": 0.5,
    }


class TestASuppliedMappingIsResolved:
    """Service mode parses what the caller declared."""

    def test_a_supplied_observation_mapping_is_parsed(self, service_policy):
        p = service_policy(observation_mapping=_MAPPING_OBS)

        assert p._obs_mapping is not None, "the observation mapping was stored and never parsed"
        assert p._obs_mapping.video == {"wrist_cam": "wrist"}
        assert p._obs_mapping.state == {"arm_joints": "arm"}

    def test_a_supplied_action_mapping_is_parsed(self, service_policy):
        p = service_policy(action_mapping=_MAPPING_ACTION)

        assert p._action_mapping is not None, "the action mapping was stored and never parsed"
        assert p._action_mapping.actions == {"arm": "joints"}

    def test_the_language_key_is_the_embodiment_key(self, service_policy):
        """Not the parser's ``"task"`` fallback, which no ``so100`` server reads.

        The fallback exists for a caller whose model declares the key, which is
        unreadable here. ``data_config`` names it for the embodiment the caller
        asked for, and it is already the key the instruction is sent under.
        """
        p = service_policy(observation_mapping=_MAPPING_OBS)

        assert p._obs_mapping is not None
        assert p._obs_mapping.language_key == _DECLARED_LANG

    def test_an_explicit_language_key_outranks_the_embodiment_key(self, service_policy):
        p = service_policy(observation_mapping=_MAPPING_OBS, language_key="instruction")

        assert p._obs_mapping is not None
        assert p._obs_mapping.language_key == "instruction"


class TestTheMappedChannelsReachTheServer:
    """The payload carries the robot's data under the model's names."""

    def test_mapped_video_and_state_are_sent(self, service_policy):
        p = service_policy(observation_mapping=_MAPPING_OBS)

        wire = p._build_service_observation(_mapped_observation(), "pick the cube")

        assert "video.wrist" in wire, "mapped camera never reached the wire"
        assert "state.arm" in wire, "mapped state never reached the wire"
        np.testing.assert_allclose(np.asarray(wire["state.arm"]).ravel(), [0.1, 0.2, 0.3])

    def test_the_payload_is_more_than_the_instruction(self, service_policy):
        """Non-vacuity: the discarded-mapping payload was the language key alone.

        Asserted as the whole key set rather than a count, so a payload that
        regains a key by some other route does not satisfy it.
        """
        p = service_policy(observation_mapping=_MAPPING_OBS)

        wire = p._build_service_observation(_mapped_observation(), "pick the cube")

        assert set(wire) == {"video.wrist", "state.arm", _DECLARED_LANG}

    def test_a_mapped_state_channel_is_float32_with_a_component_axis(self, service_policy):
        """The mapped path applies the same coercions the identity path does.

        A server rejects ``float64`` state, and a scalar reading must carry a
        component axis, so a mapping that reached the wire without these would
        be honoured into a payload the server refuses.
        """
        p = service_policy(observation_mapping={"grip": "state.gripper"})

        wire = p._build_service_observation({"grip": 0.5}, "t")

        assert wire["state.gripper"].dtype == np.float32
        assert wire["state.gripper"].shape == (1, 1)

    def test_the_mapping_does_not_change_the_payload_shape(self, service_policy):
        """A mapped payload is flat and dotted, like an unmapped one.

        The nested ``{"video": {...}, "state": {...}}`` observation is what the
        in-process Isaac-GR00T policy takes; every server version this client
        dials reads the flat form. A mapping selects *which* robot key feeds a
        wire key, so it must not decide the payload's shape - a nested payload
        would leave a mapped caller as unable to reach the model as an ignored
        mapping did.
        """
        p = service_policy(observation_mapping=_MAPPING_OBS)

        p._service_get_actions(_mapped_observation(), "pick the cube")

        assert isinstance(p._client, _RecordingClient)
        sent = p._client.sent[0]
        assert "video" not in sent and "state" not in sent, "the nested local-transport payload was sent"
        assert "video.wrist" in sent


class TestASuppliedActionMappingRenamesTheResult:
    """The returned steps use the caller's actuator names."""

    def test_the_mapped_actuator_name_is_returned(self, service_policy):
        p = service_policy(action_mapping=_MAPPING_ACTION)

        steps = p._unpack_service_actions({"action.arm": np.zeros((1, 2, 3), dtype=np.float32)})

        assert steps, "no steps unpacked"
        assert "joints" in steps[0], "the caller's actuator name is absent from the step"
        assert "arm" not in steps[0], "the bare model key was returned alongside the mapped one"

    def test_a_channel_the_mapping_omits_is_named_unmapped(self, service_policy):
        """A partial mapping loses no actuator; the surplus is visible.

        Returning the omitted channel under its bare name would make a wrong
        mapping indistinguishable from a right one at the call site.
        """
        p = service_policy(action_mapping=_MAPPING_ACTION)

        steps = p._unpack_service_actions(
            {
                "action.arm": np.zeros((1, 2, 3), dtype=np.float32),
                "action.gripper": np.zeros((1, 2, 1), dtype=np.float32),
            }
        )

        assert "joints" in steps[0]
        assert "unmapped.gripper" in steps[0]


class TestAnUnmappedCallerIsUnaffected:
    """Controls against resolving more than the caller declared."""

    def test_an_omitted_mapping_is_not_inferred(self, service_policy):
        """Inference reads the model's declared channels, which are unreadable.

        A mapping guessed here could not be validated against anything, and
        ``None`` is the signal both consumers read to mean "use the names
        ``data_config`` declares" - so acquiring one would change an unmapped
        caller's payload rather than leave it alone.
        """
        p = service_policy()

        assert p._obs_mapping is None
        assert p._action_mapping is None

    def test_the_declared_channels_still_resolve_by_name(self, service_policy):
        p = service_policy()

        wire = p._build_service_observation(_identity_observation(), "pick the cube")

        assert set(wire) == {_DECLARED_VIDEO, *_DECLARED_STATE, _DECLARED_LANG}

    def test_an_unmapped_action_chunk_returns_bare_model_keys(self, service_policy):
        p = service_policy()

        steps = p._unpack_service_actions({"action.arm": np.zeros((1, 2, 3), dtype=np.float32)})

        assert "arm" in steps[0]

    def test_a_partial_mapping_leaves_the_other_channels_resolving(self, service_policy):
        """The mapping adds the channels it names; it does not replace the set.

        The local transport sends only what the mapping names and zero-fills
        the rest, which needs the model's per-key DOF. This transport cannot
        read it, so an omitted channel keeps resolving by name rather than
        being dropped or invented: a caller who renames one channel does not
        lose the others for it.

        The mapping here is partial *within* the state modality, not merely
        across the two. A mapping that replaced its own modality's declared
        set would still leave the untouched modality resolving, so a case
        that renames only a camera cannot observe the difference: with one
        state channel renamed and its sibling not, dropping the sibling is
        what the replacing behaviour costs.
        """
        p = service_policy(observation_mapping={"arm_joints": "state.single_arm"})

        wire = p._build_service_observation(
            {
                "arm_joints": [0.0, 0.1, 0.2, 0.3, 0.4],
                "gripper": 0.5,
                "webcam": np.zeros((8, 8, 3), dtype=np.uint8),
            },
            "pick the cube",
        )

        assert set(wire) == {_DECLARED_VIDEO, *_DECLARED_STATE, _DECLARED_LANG}

    def test_the_mapping_outranks_the_identity_name_for_one_wire_key(self, service_policy):
        """When both could feed a wire key, the declared mapping wins.

        Otherwise a caller could not override a channel the robot happens to
        name as the model does, and which of the two was sent would depend on
        iteration order rather than on what was asked for.
        """
        p = service_policy(observation_mapping={"wrist_cam": "video.webcam"})

        mapped = np.full((8, 8, 3), 7, dtype=np.uint8)
        decoy = np.zeros((8, 8, 3), dtype=np.uint8)
        wire = p._build_service_observation({"wrist_cam": mapped, "webcam": decoy}, "t")

        np.testing.assert_array_equal(np.asarray(wire[_DECLARED_VIDEO]).squeeze(), mapped)
