"""Regression: a simulation backend constructor must not silently swallow a
robot-setup keyword argument.

Backend constructors accept ``**kwargs`` purely as a cross-backend
forward-compatibility sink (so one call can carry GPU-backend options like
``num_envs`` / ``device`` that non-GPU backends drop). Before this contract was
pinned, passing ``robot_name`` there - a very natural mistake, since
``robot_name`` is the ubiquitous parameter of ``run_policy`` / ``eval_policy`` /
``get_observation`` / ``send_action`` - was silently dropped. The caller got a
robot-less engine that only failed much later, and far from the cause, with an
unrelated "No world" error. That is the same "success/failure contract, wrong
effect, no signal" footgun the project already fixed for ``add_object``.

These pin the corrected contract:

* the shared ``reject_setup_kwargs`` helper raises ``TypeError`` naming the
  offending argument and pointing at the ``Robot(name, mode="sim")`` factory and
  ``add_robot``;
* genuine forward-compatibility kwargs (``num_envs`` / ``device``) are still
  tolerated and dropped;
* both the direct MuJoCo constructor and the ``create_simulation`` factory that
  forwards to it fail loudly instead of returning a robot-less engine;
* the Newton backend shares the identical contract (gated on the extra).
"""

import pytest

from strands_robots.simulation.base import reject_setup_kwargs


def test_helper_rejects_robot_name() -> None:
    with pytest.raises(TypeError) as exc:
        reject_setup_kwargs({"robot_name": "so101"})
    msg = str(exc.value)
    assert "robot_name" in msg
    # Message must be actionable: point at the factory and add_robot.
    assert "Robot(" in msg
    assert "add_robot" in msg


def test_helper_rejects_robot() -> None:
    with pytest.raises(TypeError, match="robot"):
        reject_setup_kwargs({"robot": "so101"})


def test_helper_reports_all_offending_names() -> None:
    with pytest.raises(TypeError) as exc:
        reject_setup_kwargs({"robot_name": "a", "robot": "b"})
    msg = str(exc.value)
    assert "robot_name" in msg
    assert "'robot'" in msg


def test_helper_ignores_forward_compat_kwargs() -> None:
    """Genuine backend-specific kwargs must pass through untouched (no raise)."""
    reject_setup_kwargs({"num_envs": 4, "device": "cpu"})
    reject_setup_kwargs({})


def test_mujoco_constructor_rejects_robot_name() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    with pytest.raises(TypeError) as exc:
        Simulation(robot_name="so101")
    assert "robot_name" in str(exc.value)


def test_create_simulation_factory_rejects_robot_name() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation import create_simulation

    with pytest.raises(TypeError, match="robot_name"):
        create_simulation("mujoco", robot_name="so101")


def test_mujoco_constructor_tolerates_forward_compat_kwargs() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation(num_envs=4, device="cpu", tool_name="fc_probe")
    try:
        assert sim is not None
    finally:
        sim.cleanup()


def test_newton_constructor_rejects_robot_name() -> None:
    pytest.importorskip("newton")
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    with pytest.raises(TypeError, match="robot_name"):
        NewtonSimEngine(robot_name="so101")


# --- D44: the Newton backend must reject what it does not implement ------------
#
# ``NewtonSimEngine.__init__`` ended in ``**kwargs`` documented as "Ignored;
# accepted for forward compatibility", and ``reject_setup_kwargs`` only rejects
# ``robot_name``/``robot``. So ``num_envs`` landed in the sink and was dropped
# without a word, while ``robot.py`` documented
# ``Robot("so100", backend="newton", num_envs=4096)`` as supported and both
# ``newton/__init__.py`` and ``pyproject.toml`` advertised "GPU-batched parallel
# envs". No batching code exists anywhere in the backend. Measured::
#
#     Robot(num_envs=4096) constructed: NewtonSimEngine
#     engine has num_envs attr: False
#     joint_coord_count: 6 (one arm)   world count: 1
#
# The same sink swallowed arbitrary typos (``sbsteps=3``) identically, which is
# what ``unknown_kwargs_error`` in base.py exists to prevent: a discarding sink
# "turns a misspelled or invented parameter into a successful no-op".
#
# ``num_envs`` remains a genuine ISAAC capability (``IsaacConfig.num_envs``), so
# the shared ``reject_setup_kwargs`` helper is deliberately unchanged - only the
# Newton constructor rejects it.


def test_newton_constructor_rejects_num_envs() -> None:
    """The regression: accepted, ignored, and advertised as supported."""
    pytest.importorskip("newton")
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    with pytest.raises(TypeError, match="num_envs"):
        NewtonSimEngine(num_envs=4096)


def test_the_num_envs_rejection_points_at_the_isaac_backend() -> None:
    """An unimplemented FEATURE must say what to use instead."""
    pytest.importorskip("newton")
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    with pytest.raises(TypeError) as excinfo:
        NewtonSimEngine(num_envs=4096)

    message = str(excinfo.value)
    assert "isaac" in message, message
    assert "not implemented" in message, message
    assert message.isascii()


@pytest.mark.parametrize("bad_kwarg", ["sbsteps", "nubmer_of_envs", "devcie", "num_env"])
def test_newton_constructor_rejects_an_unknown_kwarg(bad_kwarg: str) -> None:
    """A typo was indistinguishable from a working call."""
    pytest.importorskip("newton")
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    with pytest.raises(TypeError, match=bad_kwarg):
        NewtonSimEngine(**{bad_kwarg: 4})  # type: ignore[arg-type]


def test_the_unknown_kwarg_message_lists_the_accepted_names() -> None:
    pytest.importorskip("newton")
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    with pytest.raises(TypeError) as excinfo:
        NewtonSimEngine(sbsteps=3)

    message = str(excinfo.value)
    for accepted in ("solver", "substeps", "device", "nconmax", "njmax"):
        assert accepted in message, (accepted, message)
    assert message.isascii()


def test_newton_constructor_still_accepts_every_real_kwarg() -> None:
    """The rejection must not have broken the supported surface."""
    pytest.importorskip("newton")
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    engine = NewtonSimEngine(
        solver="mujoco",
        default_timestep=1.0 / 60.0,
        substeps=4,
        device=None,
        default_width=320,
        default_height=240,
        nconmax=2048,
        njmax=4096,
    )

    assert engine.substeps == 4
    assert engine.default_width == 320


def test_create_simulation_factory_rejects_num_envs_for_newton() -> None:
    """The public path an agent would actually take."""
    pytest.importorskip("newton")
    from strands_robots.simulation import create_simulation

    with pytest.raises(TypeError, match="num_envs"):
        create_simulation("newton", num_envs=4096)


def test_robot_py_no_longer_advertises_num_envs_for_newton() -> None:
    """The docstring example passed num_envs=4096 to newton; it must not.

    A doc claim that a parameter is honored is the reason the silent drop went
    unnoticed, so the claim is pinned here rather than left to drift back.
    """
    from pathlib import Path

    import strands_robots.robot as robot_module

    source = Path(robot_module.__file__).read_text()
    for line in source.splitlines():
        if 'backend="newton"' in line:
            assert "num_envs" not in line, line
