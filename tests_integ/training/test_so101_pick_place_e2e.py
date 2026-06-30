"""End-to-end SO-101 pick-and-place: the full RL stack wired together.

NOT a CI test (real MuJoCo physics, no convergence assertions in the fast suite).
Run with::

    pytest tests_integ/training/test_so101_pick_place_e2e.py -v

This is the integration proof that the four parity pieces compose on a REAL
robot + scene:

    Task 3  staged_reward     - the pick-place curriculum, authored as DATA
    (env)   SimEnv            - asymmetric actor/critic obs over the SO-101 engine
    Task 4  evaluate()        - deterministic success-rate scoring
    Task 2  GymSimEnv + SB3   - the same env consumed by stable-baselines3

It does not assert the policy SOLVES pick-place (that needs long training); it
asserts every interface connects, runs real physics, and the reward/eval/baseline
paths all execute against the actual SO-101 MuJoCo model.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

# SO-101 scene body names (verified against the registry MJCF):
#   so101/gripper                 - end-effector body
#   so101/moving_jaw_so101_v1     - the moving finger
#   cube                          - dynamic object to pick
#   target                        - static placement marker
EE_BODY = "so101/gripper"
CUBE = "cube"
TARGET = "target"

# The pick-place curriculum, authored as DATA (Task 3). No Python reward class.
PICK_PLACE_STAGES = [
    {
        # Phase 0 - Reach: pull the end-effector toward the cube.
        "reward": {"predicate": "distance_neg", "body_a": EE_BODY, "body_b": CUBE, "weight": 1.0},
        "advance_when": {"predicate": "distance_less_than", "body_a": EE_BODY, "body_b": CUBE, "threshold": 0.06},
        "bonus": 5.0,
    },
    {
        # Phase 1 - Lift/Transport: carry the cube toward the target.
        "reward": {"predicate": "distance_neg", "body_a": CUBE, "body_b": TARGET, "weight": 1.0},
        "advance_when": {"predicate": "body_above_z", "body": CUBE, "z": 0.08},
        "bonus": 10.0,
    },
    {
        # Phase 2 - Place: dense final term (terminal handled by success_fn).
        "reward": {"predicate": "distance_neg", "body_a": CUBE, "body_b": TARGET, "weight": 2.0},
    },
]

ACTOR_OBS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
    "shoulder_pan.vel",
    "shoulder_lift.vel",
    "elbow_flex.vel",
    "wrist_flex.vel",
    "wrist_roll.vel",
    "gripper.vel",
]


def _build_engine():  # type: ignore[no-untyped-def]
    import strands_robots as sr

    eng = sr.Robot("so101", mode="sim")
    eng.add_object(
        "cube", shape="box", position=[0.2, 0.0, 0.05], size=[0.02, 0.02, 0.02], color=[1, 0, 0, 1], mass=0.05
    )
    eng.add_object(
        "target", shape="box", position=[0.0, 0.2, 0.02], size=[0.03, 0.03, 0.01], color=[0, 1, 0, 1], is_static=True
    )
    return eng


def _make_env():  # type: ignore[no-untyped-def]
    from strands_robots.simulation.predicates import make_predicate
    from strands_robots.training.rl import SimEnv

    eng = _build_engine()
    staged = make_predicate("staged_reward", stages=PICK_PLACE_STAGES)
    # success: cube resting on the target marker.
    success = make_predicate("body_on", body_a=CUBE, body_b=TARGET, z_offset=0.0, xy_tol=0.06)
    return SimEnv(
        eng,
        actor_obs_keys=ACTOR_OBS,
        # Asymmetric critic: privileged cube/target-relative info would be added
        # here in a full setup; symmetric is fine for the wiring proof.
        reward_terms=[staged],
        success_fn=success,
        action_dim=6,
        max_episode_steps=40,
    )


def test_staged_reward_runs_on_so101_physics() -> None:
    """The data-authored pick-place reward computes against the real SO-101 model."""
    env = _make_env()
    obs = env.reset()
    assert obs["actor_obs"].shape == (1, 12)
    # Step with a small action; reward must be finite and the phase machine live.
    action = torch.zeros(1, 6)
    _o, reward, _done, info = env.step(action)
    assert torch.isfinite(reward).all()
    assert "terminated" in info
    # The staged term is the single reward term and exposes its phase.
    staged = env.reward_terms[0]
    assert hasattr(staged, "phase")
    assert staged.phase == 0  # starts in Reach


def test_ppo_trains_and_evaluates_on_so101() -> None:
    """Full loop: create_trainer('ppo') -> train -> evaluate, all on SO-101."""
    from strands_robots.training.rl import PpoTrainer, RLTrainSpec

    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir="/tmp/so101_ppo_e2e",
        total_timesteps=40 * 4,
        rollout_steps=40,
        num_mini_batches=4,
        num_learning_epochs=2,
        hidden_dims=(64, 64),
        seed=0,
    )
    assert trainer.validate(spec) == []
    result = trainer.train(spec)
    assert result.status == "success"
    assert result.checkpoint_dir is not None

    # Task 4: evaluate the deterministic policy; schema + success_rate present.
    metrics = trainer.evaluate(num_episodes=3)
    assert metrics["num_episodes"] == 3
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert "mean_return" in metrics


def test_gym_shim_and_sb3_on_so101() -> None:
    """Task 2: the SAME SO-101 env trains under SB3 PPO via GymSimEnv."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from strands_robots.training.rl import GymSimEnv

    env = GymSimEnv(_make_env())
    model = PPO("MlpPolicy", env, n_steps=40, batch_size=20, n_epochs=2, verbose=0, seed=0)
    model.learn(total_timesteps=80)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (6,)


def test_vectorized_ppo_on_so101_learns() -> None:
    """Task 1: vectorized PPO (N parallel SO-101 envs) collects N-batched data and improves.

    Proves the VecSimEnv path runs on REAL SO-101 MuJoCo physics, the batch is
    N*rollout_steps, and the mean episode return improves over a short run (the
    policy learns to reach toward the cube under the data-authored reward).
    """
    from strands_robots.training.rl import PpoTrainer, RLTrainSpec, VecSimEnv

    N = 4
    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=_make_env,
        output_dir="/tmp/so101_vec_e2e",
        num_envs=N,
        rollout_steps=32,
        num_mini_batches=4,
        num_learning_epochs=4,
        learning_rate=3e-3,
        init_noise_std=0.6,
        hidden_dims=(64, 64),
        seed=0,
    )
    assert trainer.validate(spec) == []
    trainer.setup(spec)
    assert isinstance(trainer.env, VecSimEnv)
    assert trainer.env.num_envs == N

    returns = []
    for _ in range(20):
        m = trainer.collect_rollout()
        trainer.update()
        returns.append(m["mean_episode_return"])
    # Batch is N * rollout_steps (the throughput multiplier).
    assert trainer._batch["actor_obs"].shape[0] == N * 32
    # Learning signal: late-window mean return beats early-window mean return.
    early = sum(returns[:5]) / 5
    late = sum(returns[-5:]) / 5
    assert late > early, f"vectorized PPO did not improve: {early:.3f} -> {late:.3f}"
    trainer.env.close()


def test_so101_reach_actually_converges() -> None:
    """Convergence proof: vectorized PPO learns to REACH the cube on SO-101.

    Stronger than the wiring tests: asserts the deterministic policy closes most
    of the gripper->cube distance after training (a real behavioural outcome,
    not just a return delta). Needs n_substeps>1 so the PD controller tracks the
    position targets - the bug the 100-cycle run surfaced.

    ~10s on CPU. Run with: pytest tests_integ/training/test_so101_pick_place_e2e.py -k reach_actually
    """
    from strands_robots.simulation.predicates import _body_position, make_predicate
    from strands_robots.training.rl import PpoTrainer, RLTrainSpec, SimEnv

    def make_reach_env():  # type: ignore[no-untyped-def]
        import strands_robots as sr

        eng = sr.Robot("so101", mode="sim")
        eng.add_object("cube", shape="box", position=[0.15, 0.10, 0.05], size=[0.02, 0.02, 0.02], mass=0.05)
        staged = make_predicate(
            "staged_reward",
            stages=[{"reward": {"predicate": "distance_neg", "body_a": EE_BODY, "body_b": CUBE, "weight": 5.0}}],
        )
        return SimEnv(
            eng,
            actor_obs_keys=ACTOR_OBS,
            reward_terms=[staged],
            action_dim=6,
            max_episode_steps=40,
            action_scale=1.5,
            n_substeps=5,
        )

    trainer = PpoTrainer()
    spec = RLTrainSpec(
        env_factory=make_reach_env,
        output_dir="/tmp/so101_reach_converge_test",
        num_envs=8,
        rollout_steps=40,
        num_mini_batches=4,
        num_learning_epochs=5,
        learning_rate=1e-3,
        init_noise_std=0.8,
        hidden_dims=(128, 128),
        gamma=0.98,
        seed=1,
    )
    trainer.setup(spec)
    returns = []
    for _ in range(100):
        m = trainer.collect_rollout()
        trainer.update()
        returns.append(m["mean_episode_return"])

    # 1. Mean return improved substantially.
    early = sum(returns[:5]) / 5
    late = sum(returns[-5:]) / 5
    assert late > early + 10.0, f"return barely moved: {early:.1f} -> {late:.1f}"

    # 2. The deterministic policy actually closes the gripper->cube distance.
    from strands_robots.training.rl import VecSimEnv

    assert isinstance(trainer.env, VecSimEnv)
    e = trainer.env.envs[0]
    e.reset()
    d0 = None
    dfin = None
    for _ in range(40):
        actor_obs = trainer._norm_actor(e._obs_dict()["actor_obs"], update=False)
        with torch.no_grad():
            action = trainer.actor_critic.act_inference(actor_obs)
        e.step(action)
        gp = _body_position(e.engine, EE_BODY)
        cp = _body_position(e.engine, CUBE)
        assert gp is not None and cp is not None
        d = ((gp[0] - cp[0]) ** 2 + (gp[1] - cp[1]) ** 2 + (gp[2] - cp[2]) ** 2) ** 0.5
        if d0 is None:
            d0 = d
        dfin = d
    # Closed at least half the initial distance.
    assert d0 is not None and dfin is not None
    assert dfin < 0.5 * d0, f"policy did not reach: {d0:.3f}m -> {dfin:.3f}m"
    trainer.env.close()
