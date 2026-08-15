"""The model key spelling a GR00T release declares must not change what resolves.

``ModalityConfig.modality_keys`` is spelled differently by GR00T release:

* N1.6 / N1.7 declare bare keys (``["front", "wrist"]``) - what
  ``examples/SO100/so100_config.py`` in Isaac-GR00T ships, and what
  ``tests_integ/groot/test_n17_live_server.py`` pins coming back over the wire.
* N1.5 declares them prefixed (``["video.front", "video.wrist"]``) - what
  ``gr00t.experiment.data_config`` ships, and what ``_load_n15`` hands to the
  N1.5 policy and then reads back.

Mapping resolution compares key *names*, so it must reduce both spellings to one
before comparing, and must state a resolved model key back in the spelling that
model declares - it is the key the payload is sent under. Against a prefixed
model the by-name arm used to match nothing, which left declaration *order* the
only thing that resolved a mapping and refused an explicit one that was correct.
"""

import pytest

msgpack = pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[groot-service]'")
zmq = pytest.importorskip("zmq", reason="zmq not installed - pip install 'strands-robots[groot-service]'")

from strands_robots.policies.groot import Gr00tPolicy  # noqa: E402
from strands_robots.policies.groot.data_config import Gr00tDataConfig, ModalityConfig  # noqa: E402
from strands_robots.policies.groot.policy import (  # noqa: E402
    _auto_infer_action_mapping,
    _auto_infer_observation_mapping,
)

# Two cameras named identically by robot and model, declared in opposite order,
# so name resolution and positional resolution give different answers.
DATA_CONFIG = Gr00tDataConfig(
    name="spelling_probe",
    video_keys=["video.wrist", "video.front"],
    state_keys=["state.single_arm", "state.gripper"],
    action_keys=["action.single_arm", "action.gripper"],
    language_keys=["annotation.human.task_description"],
    observation_indices=[0],
    action_indices=[0],
)

_MODEL_KEYS = {
    "video": ["front", "wrist"],
    "state": ["single_arm", "gripper"],
    "action": ["single_arm", "gripper"],
}


def _modality_configs(spelling: str) -> dict[str, ModalityConfig]:
    """Build model modality configs in one release's key spelling.

    Args:
        spelling: ``"bare"`` for the N1.6/N1.7 convention, ``"prefixed"`` for
            the N1.5 convention.

    Returns:
        ``{modality: ModalityConfig}`` in the requested spelling. The language
        key carries no modality prefix in either release.
    """
    configs = {
        modality: ModalityConfig(
            delta_indices=[0],
            modality_keys=[k if spelling == "bare" else f"{modality}.{k}" for k in keys],
        )
        for modality, keys in _MODEL_KEYS.items()
    }
    configs["language"] = ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"])
    return configs


class _StubLocalPolicy:
    """A loaded checkpoint that exposes only its modality configs.

    Named attributes only, so the DOF-discovery probes for ``normalizer`` and
    ``processor`` find nothing and leave ``_model_state_dof`` empty - this suite
    is about key resolution, not zero-fill sizing.
    """

    def __init__(self, modality_configs: dict[str, ModalityConfig]) -> None:
        self.modality_config = modality_configs  # N1.5 reads the singular name
        self.modality_configs = modality_configs  # N1.6 / N1.7 read the plural


def _policy(
    spelling: str,
    *,
    version: str,
    observation_mapping: dict[str, str] | None = None,
    action_mapping: dict[str, str] | None = None,
    strict_keys: bool = False,
) -> Gr00tPolicy:
    """Build a local-mode policy over a stub checkpoint and run mapping init."""
    p = Gr00tPolicy.__new__(Gr00tPolicy)
    p.data_config = DATA_CONFIG
    p.data_config_name = DATA_CONFIG.name
    p._mode = "local"
    p._groot_version = version
    p._strict = False
    p._strict_keys = strict_keys
    p._client = None
    p._local_policy = _StubLocalPolicy(_modality_configs(spelling))
    p._raw_obs_mapping = observation_mapping
    p._raw_action_mapping = action_mapping
    p._language_key_override = None
    p._obs_mapping = None
    p._action_mapping = None
    p._init_mappings()
    return p


class TestAutoInferenceResolvesByName:
    """Auto-inference must resolve by name under either spelling."""

    @pytest.mark.parametrize(
        ("spelling", "front", "wrist"),
        [
            ("bare", "front", "wrist"),
            ("prefixed", "video.front", "video.wrist"),
        ],
    )
    def test_each_camera_maps_to_the_model_key_of_the_same_name(self, spelling, front, wrist):
        mapping = _auto_infer_observation_mapping(DATA_CONFIG, _modality_configs(spelling))

        # Declared in the opposite order, so a positional resolution swaps them.
        assert mapping.video == {"wrist": wrist, "front": front}

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("bare", {"single_arm": "single_arm", "gripper": "gripper"}),
            ("prefixed", {"action.single_arm": "single_arm", "action.gripper": "gripper"}),
        ],
    )
    def test_action_keys_carry_the_spelling_the_model_declares(self, spelling, expected):
        assert _auto_infer_action_mapping(DATA_CONFIG, _modality_configs(spelling)).actions == expected

    @pytest.mark.parametrize("spelling", ["bare", "prefixed"])
    def test_a_fully_name_matching_key_set_is_not_a_strict_keys_failure(self, spelling):
        """strict_keys only refuses keys that genuinely need positional fallback."""
        mapping = _auto_infer_observation_mapping(DATA_CONFIG, _modality_configs(spelling), strict_keys=True)

        assert set(mapping.video) == {"front", "wrist"}

    @pytest.mark.parametrize("spelling", ["bare", "prefixed"])
    def test_a_genuinely_unresolvable_key_still_refuses_under_strict_keys(self, spelling):
        configs = _modality_configs(spelling)
        configs["video"] = ModalityConfig(delta_indices=[0], modality_keys=["overhead", "chest"])

        with pytest.raises(ValueError, match="cannot resolve video keys by exact name"):
            _auto_infer_observation_mapping(DATA_CONFIG, configs, strict_keys=True)


class TestASuppliedMappingIsAcceptedInEitherSpelling:
    """A caller's mapping must not depend on which release is loaded."""

    @pytest.mark.parametrize(
        ("spelling", "version", "front"),
        [
            ("bare", "n1.6", "front"),
            ("prefixed", "n1.5", "video.front"),
        ],
    )
    def test_a_correct_mapping_is_accepted_and_restated_in_the_model_spelling(self, spelling, version, front):
        p = _policy(spelling, version=version, observation_mapping={"cam": "video.front"})

        assert p._obs_mapping.video == {"cam": front}

    @pytest.mark.parametrize(
        ("spelling", "version", "action_key"),
        [
            ("bare", "n1.6", "single_arm"),
            ("prefixed", "n1.5", "action.single_arm"),
        ],
    )
    def test_a_supplied_action_key_is_restated_in_the_model_spelling(self, spelling, version, action_key):
        p = _policy(spelling, version=version, action_mapping={"action.single_arm": "joints"})

        assert p._action_mapping.actions == {action_key: "joints"}

    @pytest.mark.parametrize(
        ("spelling", "version"),
        [("bare", "n1.6"), ("prefixed", "n1.5")],
    )
    def test_a_mapping_naming_no_declared_key_is_still_refused_by_name(self, spelling, version):
        with pytest.raises(ValueError, match="model video 'wirst'"):
            _policy(spelling, version=version, observation_mapping={"cam": "video.wirst"})
