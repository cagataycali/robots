"""Regression tests for the ``base_yaw_beyond`` floating-base heading predicate.

The predicate/reward DSL grew FORWARD- and LATERAL-progress success predicates
(``base_beyond_x`` / ``base_beyond_y``) alongside the velocity-tracking reward
(``base_velocity_tracking``, which already accepts a yaw-rate ``wz`` command) and
the fall predicates (``base_tipped`` / ``base_below_z``). The YAW success half -
"the base actually turned" - was still inexpressible: ``base_beyond_x`` /
``base_beyond_y`` read only ``base_pos``, so a turn-in-place benchmark could
reward a ``wz`` command but had no way to SCORE reaching a heading goal, and
``base_tipped`` fires on ANY tilt (roll/pitch), not a deliberate turn about the
vertical.

``base_yaw_beyond(yaw, robot)`` closes that gap: TRUE when the base's world yaw
heading (extracted from ``base_quat``) has passed ``yaw`` radians (positive is a
left / counter-clockwise turn from the identity spawn). These tests set a KNOWN
base pose directly on the sim and assert the threshold, that it reads the YAW
axis (a pure roll/pitch does NOT trip it, distinguishing it from ``base_tipped``),
position/height independence, that x-position does NOT trip it (distinct from
``base_beyond_x``), live tracking, fixed-base degradation, that a goal outside the
range a heading can be measured in is refused at registration rather than scoring
every rollout the same way, and that a real ``DeclarativeBenchmark`` whose success
is ``base_yaw_beyond`` and failure is ``base_tipped`` + ``base_below_z`` succeeds
once the base turns and is vetoed if it falls. They are GL-free (``get_observation`` with ``skip_images``) so they run
in CI without a display.
"""

import logging
import math
import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    _reset_resolution_warnings,
    make_predicate,
    predicate_kind,
    register_predicate,
)

# Floating base with a NAMED free joint (a humanoid's floating_base_joint) plus
# one actuated hinge. get_observation surfaces base_quat for this robot.
NAMED_BASE_XML = """
<mujoco model="test_named_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.8 0.3 0.3 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""

# Fixed-base arm: no free joint anywhere -> no base orientation. base_yaw_beyond
# must degrade to False (and warn) rather than crash or invent a heading.
FIXED_ARM_XML = """
<mujoco model="test_fixed">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.1">
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      <joint name="j0" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="j0_act" joint="j0" kp="10"/>
  </actuator>
</mujoco>
"""


def _axis_quat(axis: str, deg: float) -> list[float]:
    """Unit (w, x, y, z) quaternion for a rotation of ``deg`` about a world axis."""
    h = math.radians(deg) / 2.0
    c, sn = math.cos(h), math.sin(h)
    return {
        "x": [c, sn, 0.0, 0.0],
        "y": [c, 0.0, sn, 0.0],
        "z": [c, 0.0, 0.0, sn],
    }[axis]


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_base_yaw_beyond", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_pose(sim, quat_wxyz: list[float] | None = None, x: float = 0.0, y: float = 0.0, z: float = 0.8) -> None:
    """Set the robot's (only) free joint to world (x, y, z) with orientation quat."""
    model, data = sim._world._model, sim._world._data
    jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qadr = int(model.jnt_qposadr[jid])
    q = quat_wxyz if quat_wxyz is not None else [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr : qadr + 7] = [x, y, z, *q]
    mujoco.mj_forward(model, data)


def test_base_yaw_beyond_is_registered_as_a_bool_predicate():
    """It must classify as bool so the DSL accepts it in success/failure clauses."""
    assert "base_yaw_beyond" in PREDICATE_REGISTRY
    assert predicate_kind("base_yaw_beyond") == "bool"


def test_base_yaw_beyond_trips_once_the_base_turns_past_the_threshold(sim):
    """FALSE while the heading is at/behind the threshold, TRUE once it passes it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=1.0)  # ~57 deg left
    _set_base_pose(sim, _axis_quat("z", 0.0))
    assert pred(sim) is False
    _set_base_pose(sim, _axis_quat("z", 50.0))  # ~0.87 rad, short of the 1.0 rad line
    assert pred(sim) is False
    _set_base_pose(sim, _axis_quat("z", 60.0))  # ~1.05 rad, past the line
    assert pred(sim) is True
    _set_base_pose(sim, _axis_quat("z", 120.0))  # turned well left
    assert pred(sim) is True


def test_base_yaw_beyond_reads_yaw_not_a_roll_or_pitch_tilt(sim):
    """It is a heading test, NOT a tilt test: a base that merely rolls or pitches
    (about a horizontal axis) has an unchanged yaw and must NOT satisfy a turn
    goal - the property that distinguishes it from base_tipped."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=0.1)  # a tiny turn goal
    for tilt in (_axis_quat("x", 80.0), _axis_quat("y", 80.0)):  # large roll / pitch
        _set_base_pose(sim, tilt)
        assert pred(sim) is False, "a pure roll/pitch tilt is not a yaw turn"
    # but a genuine turn about the vertical does satisfy it
    _set_base_pose(sim, _axis_quat("z", 30.0))
    assert pred(sim) is True


def test_base_yaw_beyond_reads_heading_not_position(sim):
    """It is distinct from base_beyond_x/y: linear displacement (with no turn)
    must NOT satisfy a heading goal, and a turn (at the origin) must NOT satisfy
    a forward-position read."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    yaw_pred = make_predicate("base_yaw_beyond", yaw=0.5)
    x_pred = make_predicate("base_beyond_x", x=1.0)
    # Walked far forward + left but never turned: heading-goal not met, x-goal met.
    _set_base_pose(sim, _axis_quat("z", 0.0), x=3.0, y=3.0)
    assert yaw_pred(sim) is False
    assert x_pred(sim) is True
    # Turned in place at the origin: heading-goal met, x-goal not met.
    _set_base_pose(sim, _axis_quat("z", 45.0), x=0.0, y=0.0)
    assert yaw_pred(sim) is True
    assert x_pred(sim) is False


def test_base_yaw_beyond_is_independent_of_position_and_height(sim):
    """It reads only the yaw heading: the same heading at any world x/y/z reads
    identically (a base that turned but drifted or dropped still counts as having
    reached the heading - the fall predicates reject a dropped base)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=1.0)
    for x, y, z in ((0.0, 0.0, 0.8), (2.5, -1.5, 0.1)):
        _set_base_pose(sim, _axis_quat("z", 0.0), x=x, y=y, z=z)
        assert pred(sim) is False, "not turned -> not beyond (any position/height)"
        _set_base_pose(sim, _axis_quat("z", 90.0), x=x, y=y, z=z)
        assert pred(sim) is True, "turned 90 deg -> beyond (any position/height)"


def test_base_yaw_beyond_tracks_the_live_base_heading(sim):
    """The predicate reads the CURRENT base heading: turning the base flips it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=0.5)
    _set_base_pose(sim, _axis_quat("z", 0.0))
    assert pred(sim) is False
    _set_base_pose(sim, _axis_quat("z", 45.0))
    assert pred(sim) is True


def test_base_yaw_beyond_accepts_a_negative_threshold(sim):
    """yaw is a SIGNED world heading (mirrors base_beyond_x/y): a negative
    threshold is a right-of-spawn heading a base at the identity spawn already
    reads True on. It is bounded to the range a heading can be measured in (see
    TestAHeadingGoalMustBeReachable) but not to one sign."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=-0.5)
    _set_base_pose(sim, _axis_quat("z", 0.0))
    assert pred(sim) is True
    _set_base_pose(sim, _axis_quat("z", -45.0))  # turned right, below -0.5 rad
    assert pred(sim) is False


def test_base_yaw_beyond_wraps_at_pi_so_a_turn_past_pi_reads_below_the_goal_again(sim):
    """The heading is atan2-wrapped to (-pi, pi], so it is single-valued only for
    a sub-pi turn - the documented reason a turn goal must stay below pi. A turn
    of just under half a revolution (170 deg -> +2.97 rad) satisfies a yaw=1.0
    goal, a turn of exactly 180 deg lands on the +pi wrap edge and still reads
    True, but turning FURTHER (190 deg) wraps the heading to -2.97 rad and the
    same goal reads False again: past pi the predicate is NOT monotonic in the
    physical turn angle. This pins that documented discontinuity so a refactor to
    a cumulative / unwrapped heading (which would keep reading True past pi) is a
    visible contract change, not a silent one."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=1.0)
    # Just under a half-revolution left: heading ~+2.97 rad, comfortably past 1.0.
    _set_base_pose(sim, _axis_quat("z", 170.0))
    assert pred(sim) is True
    # Exactly half a revolution: atan2(0, -1) = +pi, the top of the (-pi, pi] range.
    _set_base_pose(sim, _axis_quat("z", 180.0))
    assert pred(sim) is True
    # Past pi: the heading WRAPS to -2.97 rad, so the yaw=1.0 goal reads False
    # again even though the base turned FURTHER left - the wrap discontinuity.
    _set_base_pose(sim, _axis_quat("z", 190.0))
    assert pred(sim) is False
    _set_base_pose(sim, _axis_quat("z", 200.0))
    assert pred(sim) is False


def test_base_yaw_beyond_degrades_to_false_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base orientation: the predicate degrades to False
    (never turned -> never spuriously succeeds) and warns once."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_yaw_beyond", yaw=1.0)(sim)
    assert val is False
    assert any("base" in r.message.lower() for r in caplog.records)


def test_declarative_turn_benchmark_succeeds_on_turn_and_is_vetoed_by_a_fall(sim):
    """End to end: a DeclarativeBenchmark whose success is base_yaw_beyond and
    whose failure is base_tipped + base_below_z - the yaw velocity-tracking task
    vocabulary (tracking reward with a wz command shapes HOW to turn, fall
    predicates end a bad rollout, base_yaw_beyond scores the GOAL) - compiles and
    reports success only once the base turns past the heading, never while it is
    standing un-turned, and a topple vetoes the run via the failure clause."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    bench = DeclarativeBenchmark.from_dict(
        {
            "name": "turn-left",
            "default_robot": "humanoid",
            "max_steps": 1000,
            "dense_reward": [
                {"predicate": "base_velocity_tracking", "wz": 0.5, "lin_weight": 1.0},
            ],
            "success": {"all": [{"predicate": "base_yaw_beyond", "yaw": 1.0}]},
            "failure": {
                "any": [
                    {"predicate": "base_tipped", "tol": 0.7},
                    {"predicate": "base_below_z", "z": 0.3},
                ]
            },
        }
    )
    # Standing upright, not turned: not fallen, but has NOT reached the heading.
    _set_base_pose(sim, _axis_quat("z", 0.0), z=0.8)
    assert bench.is_failure(sim) is False
    assert bench.is_success(sim) is False, "standing un-turned must not score the turn goal"
    # Turned past the line, still upright: the goal is reached.
    _set_base_pose(sim, _axis_quat("z", 90.0), z=0.8)
    assert bench.is_failure(sim) is False
    assert bench.is_success(sim) is True
    # Toppled (pitched onto its side): the fall predicate fires and vetoes the
    # rollout (a toppled base's yaw is ill-defined, so the tilt is the terminal).
    _set_base_pose(sim, _axis_quat("y", 90.0), z=0.8)
    assert bench.is_failure(sim) is True


# Headings the base can actually report. ``atan2`` returns a value in (-pi, pi],
# so this samples that whole interval - it is the set every case below reads its
# verdict from, which is what makes "no heading satisfies this goal" a measurement
# rather than an argument about the formula.
_MEASURABLE_HEADINGS = [
    -math.pi + 1e-9,  # the open end: a heading arbitrarily close to -pi, never -pi
    *(-math.pi + i * (2.0 * math.pi) / 240.0 for i in range(1, 240)),
    math.pi,  # the closed end: atan2(0, -1) really does report +pi
]

# Goals no measurable heading discriminates, and why each one is reached.
_UNREACHABLE_GOALS = [
    pytest.param(180.0, id="180-written-as-degrees"),
    pytest.param(90.0, id="90-written-as-degrees"),
    pytest.param(math.degrees(1.0), id="one-radian-converted-to-degrees"),
    pytest.param(math.pi, id="exactly-plus-pi"),
    pytest.param(-math.pi, id="exactly-minus-pi"),
    pytest.param(-4.0, id="past-minus-pi"),
    pytest.param(2.0 * math.pi, id="a-full-revolution"),
]

# Goals that do discriminate, including both boundaries.
_REACHABLE_GOALS = [
    pytest.param(1.0, id="the-shipped-go2-turn-left-goal"),
    pytest.param(0.0, id="any-left-turn-at-all"),
    pytest.param(-1.0, id="a-right-of-spawn-heading"),
    pytest.param(math.pi - 1e-6, id="just-inside-plus-pi"),
    pytest.param(-math.pi + 1e-6, id="just-inside-minus-pi"),
]


def _discriminates(goal: float) -> bool:
    """Whether SOME measurable heading meets ``goal`` and some other does not.

    A goal that fails this decides the clause before the rollout starts: the
    success verdict is the same for every orientation the base can reach, so the
    clause reports on the goal rather than on the robot. This is the property the
    domain exists to protect, stated here in the test rather than as the
    implementation's ``abs(yaw) >= pi`` - which is why moving that bound in either
    direction is caught below.
    """
    met = [h > goal for h in _MEASURABLE_HEADINGS]
    return any(met) and not all(met)


class TestAHeadingGoalMustBeReachable:
    """A turn goal outside the measurable heading range is refused, not compiled.

    The heading is read from ``base_quat`` through ``atan2``, which only ever
    reports an angle in ``(-pi, pi]``. A goal at or above ``+pi`` is therefore met
    by no orientation the base can reach and a goal at or below ``-pi`` is met by
    every one - so the success clause is decided before the rollout starts, under
    ``status="success"`` at registration. Measured on the pre-fix tree with a go2
    posed at 61 headings spanning 0..pi: ``yaw=1.0`` read True at 41 of them,
    ``yaw=57.3`` (1 rad written as degrees) and ``yaw=180`` at 0, and ``yaw=-4.0``
    at all 61 including the spawn pose - all four accepted, none refused. Degrees
    is the route that matters: the docstring quotes the goal as "~57 deg", so the
    unit is in the author's hands and the wrong one compiles clean.

    This is the same permanently-decided clause a negative tolerance produces (see
    ``test_predicate_tolerance_sign_domain``), reached by a different route, and it
    is refused at the same choke point.
    """

    def test_the_sweep_reads_the_range_the_predicate_can_report(self, sim):
        """Non-vacuity: the sampled headings really are what the base reports.

        Every verdict below rests on :data:`_MEASURABLE_HEADINGS` covering the
        ``atan2`` range, so that is measured against the live predicate instead of
        assumed - a sample set that missed the interval would make the
        discrimination cases vacuous.
        """
        sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
        pred = make_predicate("base_yaw_beyond", yaw=0.0)
        _set_base_pose(sim, _axis_quat("z", -90.0))
        assert pred(sim) is False
        _set_base_pose(sim, _axis_quat("z", 90.0))
        assert pred(sim) is True
        assert min(_MEASURABLE_HEADINGS) > -math.pi, "the range is open at -pi"
        assert max(_MEASURABLE_HEADINGS) == math.pi, "and closed at +pi"
        _set_base_pose(sim, _axis_quat("z", 180.0))
        assert make_predicate("base_yaw_beyond", yaw=math.pi - 1e-6)(sim) is True, (
            "a half-revolution reports +pi, so the closed end is reachable"
        )

    @pytest.mark.parametrize("goal", _UNREACHABLE_GOALS)
    def test_a_goal_no_heading_discriminates_is_refused(self, goal):
        with pytest.raises(ValueError) as excinfo:
            make_predicate("base_yaw_beyond", yaw=goal)
        message = str(excinfo.value)
        assert "base_yaw_beyond" in message
        assert "yaw" in message
        assert "(-pi, pi)" in message
        assert repr(goal) in message

    @pytest.mark.parametrize("goal", _REACHABLE_GOALS)
    def test_a_goal_some_heading_discriminates_is_accepted(self, goal):
        assert callable(make_predicate("base_yaw_beyond", yaw=goal))

    @pytest.mark.parametrize("goal", _UNREACHABLE_GOALS)
    def test_the_refused_goals_are_exactly_the_ones_no_heading_discriminates(self, goal):
        """The bound is pinned by the property, not by the number pi.

        Widening it (to ``2 * pi``, say) admits a goal this asserts nothing can
        discriminate; narrowing it (to ``1.0``) refuses one the sibling case above
        asserts is usable. Both cases have to hold for the bound to be right.
        """
        assert not _discriminates(goal)

    @pytest.mark.parametrize("goal", _REACHABLE_GOALS)
    def test_the_accepted_goals_are_all_discriminating(self, goal):
        assert _discriminates(goal)

    def test_a_refused_goal_would_have_scored_every_pose_alike(self, sim):
        """Why it is refused, in the units of the predicate itself.

        Read through the live predicate rather than the formula: the factory is
        called past the domain guard so the pre-fix behaviour is exercised on a
        real posed base, and the verdict is the same at a heading the goal was
        meant to reject and at one it was meant to accept.
        """
        sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
        degrees_goal = PREDICATE_REGISTRY["base_yaw_beyond"](yaw=180.0)
        reachable_goal = make_predicate("base_yaw_beyond", yaw=1.0)
        for heading_deg in (0.0, 60.0, 170.0):
            _set_base_pose(sim, _axis_quat("z", heading_deg))
            assert degrees_goal(sim) is False, "no reachable heading meets a degrees-spelled goal"
        _set_base_pose(sim, _axis_quat("z", 0.0))
        assert reachable_goal(sim) is False
        _set_base_pose(sim, _axis_quat("z", 60.0))
        assert reachable_goal(sim) is True

    def test_a_declarative_benchmark_is_refused_at_compile_not_at_rollout(self, sim):
        """The refusal reaches the surface a benchmark author actually writes.

        ``DeclarativeBenchmark`` compiles its clauses through ``make_predicate``,
        so the goal is refused while the spec is being read - not by a rollout that
        burns its whole step budget reporting an honest miss.
        """
        sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
        spec = {
            "name": "turn-left-in-degrees",
            "default_robot": "humanoid",
            "max_steps": 1000,
            "success": {"all": [{"predicate": "base_yaw_beyond", "yaw": 180}]},
            "failure": {"any": [{"predicate": "base_tipped", "tol": 0.7}]},
        }
        with pytest.raises(ValueError, match=r"\(-pi, pi\)"):
            DeclarativeBenchmark.from_dict(spec)
        # The same spec in radians compiles and scores the turn it was written for.
        spec["success"] = {"all": [{"predicate": "base_yaw_beyond", "yaw": math.radians(90.0)}]}
        bench = DeclarativeBenchmark.from_dict(spec)
        _set_base_pose(sim, _axis_quat("z", 30.0))
        assert bench.is_success(sim) is False
        _set_base_pose(sim, _axis_quat("z", 120.0))
        assert bench.is_success(sim) is True

    def test_a_heading_on_a_later_registered_predicate_is_covered(self):
        """The domain is read from the param name, so it needs no registry edit."""

        def _factory(yaw=0.0):
            def check(_sim):
                return False

            return check

        # The domain is read from ``__annotations__`` by param name, and this module
        # does not postpone annotation evaluation, so a literal ``yaw: float`` here
        # would store the ``float`` CLASS where the guard reads the string form.
        # Declared the way the module sees a shipped factory instead.
        _factory.__annotations__["yaw"] = "float"
        register_predicate("probe_heading_domain", _factory)
        try:
            with pytest.raises(ValueError, match="yaw"):
                make_predicate("probe_heading_domain", yaw=180.0)
            assert callable(make_predicate("probe_heading_domain", yaw=1.0))
        finally:
            PREDICATE_REGISTRY.pop("probe_heading_domain", None)

    def test_a_non_finite_goal_still_reports_finiteness(self):
        """The pre-existing reason is not displaced by the new one.

        ``nan`` is not comparable to pi, so the heading check must sit behind the
        finiteness guard whose coercion it depends on.
        """
        for value in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="finite") as excinfo:
                make_predicate("base_yaw_beyond", yaw=value)
            assert "(-pi, pi)" not in str(excinfo.value)
