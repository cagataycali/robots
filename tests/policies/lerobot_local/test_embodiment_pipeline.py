"""Real-pipeline injection tests for the declarative embodiment mapping.

Unlike the mock-heavy existing suite (which let B7/B12 slip past), these load an
ACTUAL LeRobot processor pipeline and verify the embodiment map is injected and
transforms raw strands-native observations correctly. Skips cleanly if the
model config isn't cached / lerobot processor unavailable.

Hub offline mode is forced for the whole module (see ``_force_hub_offline``):
``ProcessorBridge.from_pretrained`` loads from the local cache or fails fast and
skips, so a slow or unreachable Hub can never trigger an unbounded background
download that hangs the suite past ``pytest-timeout``.
"""

import numpy as np
import pytest

pytest.importorskip("lerobot")

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.processor import ProcessorBridge

SMOLVLA = "lerobot/smolvla_base"


@pytest.fixture(autouse=True)
def _force_hub_offline(monkeypatch):
    """Pin huggingface_hub to offline mode for every test in this module.

    ``ProcessorBridge.from_pretrained`` reads a real checkpoint from the Hub.
    With a reachable cache that is fast; with a slow/unreachable Hub the xet
    download neither completes nor raises promptly, so the ``try/except`` skip
    guard in ``_load_bridge`` never fires and the whole session is killed by
    ``pytest-timeout``. Forcing offline makes the load deterministic: a cache
    hit runs the real pipeline, a cache miss raises ``LocalEntryNotFoundError``
    immediately and the test skips.

    ``huggingface_hub.constants.HF_HUB_OFFLINE`` is evaluated once at import,
    so the environment variable alone is insufficient once the package is
    imported; the module global must be patched directly. ``is_offline_mode()``
    reads that global at call time, so the patch takes effect immediately.
    """
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _load_bridge():
    """Load SmolVLA's real preprocessor; skip if unavailable/uncached."""
    bridge = None
    try:
        bridge = ProcessorBridge.from_pretrained(
            SMOLVLA,
            device="cpu",
            policy_type="smolvla",
            overrides={"device_processor": {"device": "cpu"}},
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SmolVLA pipeline unavailable: {exc}")
    if bridge is None or not bridge.has_preprocessor:
        pytest.skip("SmolVLA preprocessor not loaded")
    return bridge


def test_load_bridge_runs_under_hub_offline_mode(monkeypatch):
    """The bridge load must run with the Hub in offline mode.

    Regression for a CI hang: ``from_pretrained`` triggered an unbounded xet
    download that timed out the whole suite instead of skipping. The fix pins
    the module to offline mode so the load is cache-only. This asserts the
    invariant at the exact call site by spying on ``from_pretrained`` and
    checking ``is_offline_mode()`` is active when it runs. Fails before the fix
    (offline not forced -> ``is_offline_mode()`` is False).
    """
    import huggingface_hub.constants as hf_constants

    seen = {}

    def spy(*_args, **_kwargs):
        seen["offline"] = hf_constants.is_offline_mode()
        raise RuntimeError("forced: do not actually load weights")

    monkeypatch.setattr(ProcessorBridge, "from_pretrained", spy)

    with pytest.raises(pytest.skip.Exception):
        _load_bridge()

    assert seen.get("offline") is True, "from_pretrained must run under HF Hub offline mode"


def test_apply_embodiment_injects_rename_and_pack():
    bridge = _load_bridge()
    pre = bridge._preprocessor
    em = EmbodimentMap(
        name="t",
        obs_rename={"front": "observation.images.top", "wrist": "observation.images.wrist"},
        state_keys=["1", "2", "3", "4", "5", "6"],
        action_keys=["1", "2", "3", "4", "5", "6"],
        dim_policy="pad",
    )
    bridge.apply_embodiment(em, input_features={})
    names = [getattr(s, "_registry_name", type(s).__name__) for s in pre.steps]
    # rename first, pack-state immediately after
    assert names[0] == "rename_observations_processor"
    assert names[1] == "strands_pack_state"
    assert pre.steps[0].rename_map == em.obs_rename


def test_apply_embodiment_idempotent():
    bridge = _load_bridge()
    pre = bridge._preprocessor
    em = EmbodimentMap(name="t", obs_rename={}, state_keys=["1", "2"], dim_policy="pad")
    bridge.apply_embodiment(em, input_features={})
    bridge.apply_embodiment(em, input_features={})  # re-apply
    names = [getattr(s, "_registry_name", type(s).__name__) for s in pre.steps]
    assert names.count("strands_pack_state") == 1


def test_raw_obs_transforms_through_injected_steps():
    """RAW sim obs -> rename + pack steps -> LeRobot keys, no strands remap."""
    from lerobot.processor import TransitionKey
    from lerobot.processor.converters import create_transition

    bridge = _load_bridge()
    pre = bridge._preprocessor
    em = EmbodimentMap(
        name="t",
        obs_rename={"front": "observation.images.top", "wrist": "observation.images.wrist"},
        state_keys=["1", "2", "3", "4", "5", "6"],
        dim_policy="pad",
    )
    bridge.apply_embodiment(em, input_features={})

    raw = {
        "front": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist": np.zeros((4, 4, 3), dtype=np.uint8),
        "1": 0.1,
        "2": 0.2,
        "3": 0.3,
        "4": 0.4,
        "5": 0.5,
        "6": 0.6,
    }
    t = create_transition(observation=raw, complementary_data={"task": "pick"})
    t = pre.steps[0](t)  # rename
    t = pre.steps[1](t)  # pack-state
    obs = t[TransitionKey.OBSERVATION]

    assert "observation.images.top" in obs
    assert "observation.images.wrist" in obs
    assert "observation.state" in obs
    assert list(np.asarray(obs["observation.state"]).ravel()) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # raw strands keys are gone
    assert "front" not in obs and "1" not in obs
