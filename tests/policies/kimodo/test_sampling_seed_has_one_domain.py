"""One Kimodo sampling seed, one domain, whichever surface sets it.

The seed is used twice, and the two uses have to agree. It is handed to the
sampler, and it is part of the key ``KimodoPolicy`` identifies the buffered
motion by - a key built by coercing the seed with ``int()``. A seed that is not
already whole therefore names a different sample than the one it produces:
``2.5`` and ``2.9`` reach the sampler as themselves and key as ``2``, so the
second reseed reads as a cache hit and replays the first episode's motion while
reporting that a new seed was applied. ``nan`` and ``inf`` do not survive the
coercion at all and raise out of the private key builder mid-rollout, and
``inf`` arrives from a JSON config file as the well-formed ``1e400``.

Four surfaces set this seed - the constructor, ``from_dict``, ``from_json``, and
a ``KimodoPolicy(seed=...)`` override - and a fifth, ``KimodoPolicy.reset``,
stores a per-episode reseed with ``object.__setattr__`` and so never re-enters
``__post_init__``. All five consult ``sampling_seed_error``, so the tables below
run every spelling past every surface.

The ACCEPTED rows are controls that hold on both sides of the fix: a whole seed
of either sign or any width still works, ``2`` and ``2.0`` still deliberately
name ONE motion, and ``None`` still draws fresh entropy. Magnitude is left to
the appliers, which report it themselves.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from strands_robots.policies.kimodo import KIMODO_G1_JOINTS, KimodoConfig, KimodoPolicy

_NUM_JOINTS = len(KIMODO_G1_JOINTS)
_ROOT = 7

# One native frame per emitted frame, so a sampler run is countable directly.
_FAST = {"num_frames": 6, "native_fps": 30, "tracker_fps": 30}


class _SeedRecordingAgent:
    """Return a motion determined by the seed, and record every seed received."""

    def __init__(self) -> None:
        self.seeds: list[object] = []

    def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
        self.seeds.append(seed)
        out = np.zeros((num_frames, _ROOT + _NUM_JOINTS), dtype=np.float32)
        out[:, 3] = 1.0  # identity quaternion, wxyz
        # A distinct offset per seed, so a replayed motion is distinguishable
        # from a freshly sampled one without inspecting call counts alone, plus
        # a per-frame ramp so a rewind is observable too. Not a seeded RNG:
        # ``numpy.random.default_rng`` refuses the negative seed this domain
        # deliberately keeps, which would make the control untestable.
        offset = 0.0 if seed is None else float(int(seed) % 1000) / 1000.0
        out[:, _ROOT:] = offset + 0.01 * np.arange(num_frames, dtype=np.float32)[:, None]
        return out


def _policy(**cfg_kwargs):
    agent = _SeedRecordingAgent()
    return KimodoPolicy(config=KimodoConfig(**_FAST, **cfg_kwargs), motion_agent=agent), agent


def _pose(policy) -> tuple[float, ...]:
    obs = {"state": np.zeros(_NUM_JOINTS, dtype=np.float32)}
    action = asyncio.run(policy.get_actions(obs, "walk forward"))[0]
    return tuple(action[joint] for joint in KIMODO_G1_JOINTS)


#: Seeds that cannot key the sample they produce. Each is refused by every
#: surface that sets a seed.
UNUSABLE = [
    pytest.param(2.5, id="fractional-float"),
    pytest.param(2.9, id="fractional-float-rounding-to-the-same-whole"),
    pytest.param(True, id="bool-an-int-subclass-that-would-key-as-1"),
    pytest.param("2", id="numeric-string"),
    pytest.param("walk forward", id="non-numeric-string"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(json.loads("1e400"), id="1e400-from-a-json-config-file"),
]

#: Seeds that survive both uses. Controls: green before and after the fix.
USABLE = [
    pytest.param(None, id="None-draws-fresh-entropy"),
    pytest.param(0, id="zero"),
    pytest.param(2, id="int"),
    pytest.param(2.0, id="integral-float"),
    pytest.param(-1, id="negative-torch-honors-it"),
    pytest.param(10**400, id="wider-than-the-applier-which-reports-it-itself"),
]


class TestEverySurfaceThatSetsTheSeedSharesOneDomain:
    """A seed gets one verdict whichever of the six surfaces receives it."""

    @pytest.mark.parametrize("seed", UNUSABLE)
    def test_the_constructor_refuses_it(self, seed):
        with pytest.raises(ValueError, match="seed must be a whole number or None"):
            KimodoConfig(seed=seed)

    @pytest.mark.parametrize("seed", UNUSABLE)
    def test_from_dict_refuses_it(self, seed):
        with pytest.raises(ValueError, match="seed must be a whole number or None"):
            KimodoConfig.from_dict({"seed": seed})

    @pytest.mark.parametrize("seed", UNUSABLE)
    def test_a_policy_level_override_refuses_it(self, seed):
        with pytest.raises(ValueError, match="seed must be a whole number or None"):
            KimodoPolicy(seed=seed, motion_agent=_SeedRecordingAgent())

    @pytest.mark.parametrize("seed", UNUSABLE)
    def test_a_per_episode_reseed_refuses_it(self, seed):
        """``reset`` writes past the frozen dataclass, so it applies the domain itself."""
        policy, _ = _policy()
        with pytest.raises(ValueError, match="seed must be a whole number or None"):
            policy.reset(seed=seed)

    def test_a_json_config_file_refuses_the_infinity_it_can_spell(self, tmp_path):
        """``1e400`` is well-formed JSON and parses to ``inf``."""
        path = tmp_path / "kimodo.json"
        path.write_text('{"seed": 1e400}', encoding="utf-8")
        with pytest.raises(ValueError, match="seed must be a whole number or None"):
            KimodoConfig.from_json(path)

    @pytest.mark.parametrize("seed", UNUSABLE)
    def test_a_per_call_seed_override_in_get_actions_refuses_it(self, seed):
        """The per-call seed kwarg on get_actions also consults the domain."""
        policy, _ = _policy(seed=42)
        with pytest.raises(ValueError, match="seed must be a whole number or None"):
            asyncio.run(policy.get_actions({}, "waving", seed=seed))

    @pytest.mark.parametrize("seed", UNUSABLE)
    def test_all_three_surfaces_state_the_same_domain_and_name_themselves(self, seed):
        """One domain, and each message says which surface refused, so a caller
        knows what to change and where."""
        with pytest.raises(ValueError) as from_construction:
            KimodoConfig(seed=seed)
        policy, _ = _policy()
        with pytest.raises(ValueError) as from_reseed:
            policy.reset(seed=seed)
        with pytest.raises(ValueError) as from_get_actions:
            asyncio.run(policy.get_actions({}, "waving", seed=seed))

        domain = "seed must be a whole number or None"
        assert str(from_construction.value).startswith(f"KimodoConfig: {domain}")
        assert str(from_reseed.value).startswith(f"KimodoPolicy.reset: {domain}")
        assert str(from_get_actions.value).startswith(f"KimodoPolicy.get_actions: {domain}")


class TestARefusedSeedChangesNothing:
    """A refusal is not a partial application."""

    def test_a_refused_reseed_leaves_the_held_motion_and_the_cursor_alone(self):
        policy, agent = _policy(seed=7)
        first = _pose(policy)
        second = _pose(policy)
        assert first != second, "the stub motion must advance, or a rewind is unobservable"

        with pytest.raises(ValueError, match="seed must be a whole number or None"):
            policy.reset(seed=float("inf"))

        assert policy.config.seed == 7, "the frozen config must not hold a seed that was refused"
        assert len(agent.seeds) == 1, "a refused reseed must not run the sampler"
        assert _pose(policy) != first, "the cursor must not have been rewound"


class TestASeedTheDomainAcceptsSurvivesBothOfItsUses:
    """The controls: every accepted seed still reaches the sampler and keys itself."""

    @pytest.mark.parametrize("seed", USABLE)
    def test_it_is_accepted_by_every_surface(self, seed):
        assert KimodoConfig(seed=seed).seed == seed
        assert KimodoConfig.from_dict({"seed": seed}).seed == seed
        policy, _ = _policy()
        policy.reset(seed=seed)

    @pytest.mark.parametrize("seed", USABLE)
    def test_it_keys_the_sample_it_produces(self, seed):
        """The key holds the seed itself, so it names the run the sampler made."""
        keyed = KimodoPolicy._sample_key("walk forward", 100, 7.5, seed)[3]
        assert keyed == seed

    def test_two_distinct_accepted_seeds_name_two_motions(self):
        """The invariant a fractional seed broke: distinct seeds, distinct samples."""
        policy, agent = _policy(seed=2)
        first = _pose(policy)

        policy.reset(seed=9)
        after = _pose(policy)

        assert agent.seeds == [2, 9], "the second seed must reach the sampler"
        assert first != after, "episode two must not replay episode one's motion"

    def test_an_integral_float_still_names_the_same_motion_as_the_whole_number(self):
        """``2`` and ``2.0`` seed the sampler identically, so they share one key."""
        assert KimodoPolicy._sample_key("walk forward", 100, 7.5, 2.0) == KimodoPolicy._sample_key(
            "walk forward", 100, 7.5, 2
        )

        policy, agent = _policy(seed=2)
        _pose(policy)
        policy.reset(seed=2.0)
        _pose(policy)
        assert len(agent.seeds) == 1, "the same seed spelled two ways must not re-run the sampler"

    def test_none_still_draws_a_fresh_sample_rather_than_seeding(self):
        policy, agent = _policy()
        _pose(policy)
        assert agent.seeds == [None]

    def test_reset_without_a_seed_still_rewinds_without_re_sampling(self):
        policy, agent = _policy(seed=3)
        first = _pose(policy)
        _pose(policy)

        policy.reset()

        assert _pose(policy) == first
        assert len(agent.seeds) == 1
