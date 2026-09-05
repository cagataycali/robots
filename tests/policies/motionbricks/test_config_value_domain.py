"""Value-domain contracts for :class:`MotionBricksConfig`'s knobs and its path.

``__post_init__`` has always *had* a check for ``fps`` and ``generate_dt``. What
it had was a COMPARISON, and a comparison is not a domain: ``nan < 1`` and
``nan <= 0`` are both ``False``, ``inf < 1`` is ``False``, and ``True`` is an
``int`` subclass that satisfies ``>= 1``. So the guard that exists to make "an
out-of-range synthesis knob surface at construction with a clear message rather
than as an opaque failure deep inside the generator" (this module's own
docstring) let every one of those through.

They do not stop at the config. All three knobs are read by
:attr:`MotionBricksConfig.controller_dt` -
``(NUM_REGEN_FRAMES / fps) * generate_dt`` - which
:class:`MotionBricksPolicy` caches as ``self._controller_dt`` and passes to
``MotionAgent.next_qpos`` on every ``get_actions`` call, where the real adapter
hands it straight to the upstream ``full_navigation_agent.generate_new_frames``.
Measured on the pre-fix tree, per accepted value:

===================== ===================== =====================================
value                 ``controller_dt``     what the generator was asked for
===================== ===================== =====================================
``fps=float("nan")``  ``nan``               a horizon that poisons every frame
``fps=float("inf")``  ``0.0``               integrate nothing, under success
``fps=True``          ``16.0``              a 1 Hz frame rate from a boolean
``fps=2.5``           ``3.2``               a fractional frame rate
``generate_dt=nan``   ``nan``               as above
``generate_dt=inf``   ``inf``               an unbounded horizon
``generate_dt=True``  ``0.2666...``         a multiplier from a boolean
===================== ===================== =====================================

``speed_scale`` had a second failure of its own: the pair was normalised with
``float(scale[0]), float(scale[1])`` and only *then* range-checked, so
``("1", "2")`` was laundered into ``(1.0, 2.0)`` and stored as if the caller had
written floats, while ``(nan, nan)`` and ``(1.0, inf)`` passed the
``lo <= 0 or hi <= 0 or hi < lo`` test unchanged and were stored verbatim.

The fix composes the shared domains in :mod:`strands_robots.utils` -
:func:`~strands_robots.utils.positive_whole_number_error` for ``fps`` (whose own
docstring already names a recorder's ``fps`` as a caller) and
:func:`~strands_robots.utils.positive_finite_number_error` for ``generate_dt``
and each ``speed_scale`` component - so the refusal text is identical to every
other rate and multiplier in the library rather than merely equivalent in
verdict. These tests pin that domain through each public surface a caller
reaches (the constructor, ``from_dict``, ``from_file``), pin the consequence the
domain exists to prevent, and pin the values that stay first-class so the guard
cannot creep into refusing a generator a caller may legitimately ask for.

The paths are the second axis, and they are a different question rather than a
smaller one: the values they must refuse are not out of range, they are not
paths. ``result_dir``'s check was ``if not self.result_dir``, which asserts
truthiness - so ``123`` and ``["out"]`` were accepted for being truthy and ``0``
refused for being falsy, by a message about an empty *path*, about a number. The
values sorted by a property the field does not have. Its domain is now "a value a
path can be read from" (``str`` or :class:`os.PathLike`), normalised to the
``str`` the field declares, and ``TestResultDirPathDomain`` pins that alongside
the two consequences the truthiness test carried beyond the eventual
``Path(...)``: a ``Path`` was stored unnormalised, so a config built from one
compared unequal to the identical config built from a ``str``, and a list left
this frozen - therefore hashable - dataclass unhashable.

``skeleton_xml`` and ``scene_xml`` name locations too, and they were the two
fields that axis skipped: of the eleven declared fields they were the only ones
``__post_init__`` did not mention at all, while the roster of what is
deliberately out of scope (``TestNeighbouringConfigFieldsStayOutOfScope``) lists
the three enumerations and not them. Every row of the ``result_dir`` table above
reproduced on them unchanged, and a falsy value fared *worse* rather than
better: ``result_dir=0`` was at least refused, whereas ``skeleton_xml=0`` and
``skeleton_xml=""`` were accepted and then discarded by ``config.skeleton_xml or
<derived default>`` in ``_MotionBricksAgentAdapter.build``, so the generator was
built against the derived skeleton and the run reported success. That is the one
value ``None`` already means, and it is the reason the empty string is refused
here rather than admitted as a second spelling of it.
``TestOptionalPathFieldsShareThatDomain`` pins the axis on both fields;
``test_the_skeleton_the_caller_named_is_the_skeleton_the_generator_is_built_with``
in ``test_agent_adapter_build.py`` pins the consequence at the generator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.motionbricks import MotionBricksConfig, MotionBricksPolicy
from strands_robots.policies.motionbricks.config import NUM_REGEN_FRAMES
from strands_robots.policies.motionbricks.observation import resolve_mode

# Values no numeric knob of this config can be honored as, whatever its sign
# rule: a non-finite one poisons or collapses the horizon it scales, a boolean
# would act as a silent 1, and a non-real one has no horizon at all.
UNUSABLE_ANY_KNOB: list[Any] = [
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "2.0",
    None,
    [2.0],
    10**400,
]


def _config(**kwargs: Any) -> MotionBricksConfig:
    """Build a config through one funnel.

    These tests deliberately supply values outside the declared field types (a
    string where a ``float`` is annotated, a list where a scalar is), which is
    the point - the runtime is what must refuse them. Splatting through one
    ``**kwargs: Any`` funnel states that intent once instead of scattering a
    suppression over every call.
    """
    return MotionBricksConfig(**{"result_dir": "out", **kwargs})


def _horizon(*, fps: Any, generate_dt: Any) -> float:
    """The horizon a value WOULD have produced, out of the real property.

    Applied to a ``SimpleNamespace`` rather than to a config, because a config
    carrying these values can no longer be built - that is the fix. Reading
    ``MotionBricksConfig.controller_dt`` itself (not a re-derivation of its
    arithmetic) keeps these measurements honest if the formula ever changes.
    """
    return float(MotionBricksConfig.controller_dt.fget(SimpleNamespace(fps=fps, generate_dt=generate_dt)))  # type: ignore[attr-defined]


class _RecordingAgent:
    """Records the ``controller_dt`` the policy forwards per ``get_actions``.

    Stands in for the ``motion_agent`` seam so the horizon can be observed
    reaching the generator without the ``motionbricks`` install or a GPU.
    """

    clip_keys = ["idle", "walk"]
    clip_token_specs: list[list[int] | None] = [None, [1] * 11]
    min_token = 6
    max_token = 16

    def __init__(self) -> None:
        self.horizons: list[float] = []

    def reset(self) -> None:
        pass

    def next_qpos(self, control_signals: dict[str, Any], controller_dt: float) -> np.ndarray:
        self.horizons.append(controller_dt)
        return np.zeros(7 + 29, dtype=np.float64)


class TestFpsDomain:
    """``fps`` is the divisor of the generator's integration horizon."""

    @pytest.mark.parametrize("value", [0, -30, 2.5, *UNUSABLE_ANY_KNOB])
    def test_an_unusable_fps_is_refused_at_construction(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"fps"):
            _config(fps=value)

    @pytest.mark.parametrize("value", [1, 30, 60, 30.0, np.int64(30), np.float64(50)])
    def test_a_usable_fps_still_builds(self, value: Any) -> None:
        assert _config(fps=value).controller_dt == pytest.approx(NUM_REGEN_FRAMES / float(value) * 2.0)

    def test_the_refusal_names_the_field_and_the_value(self) -> None:
        with pytest.raises(ValueError, match=r"MotionBricksConfig: fps must be a positive whole number, got 0\."):
            _config(fps=0)

    def test_a_non_real_fps_is_refused_rather_than_raising_from_the_comparison(self) -> None:
        # Pre-fix this escaped as ``'<' not supported between instances of 'str'
        # and 'int'``, which names neither the field nor the config.
        with pytest.raises(ValueError, match=r"MotionBricksConfig: fps must be a positive whole number, got '30'\."):
            _config(fps="30")


class TestGenerateDtDomain:
    """``generate_dt`` is the multiplier of that same horizon."""

    @pytest.mark.parametrize("value", [0, 0.0, -2.0, *UNUSABLE_ANY_KNOB])
    def test_an_unusable_generate_dt_is_refused_at_construction(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"generate_dt"):
            _config(generate_dt=value)

    @pytest.mark.parametrize("value", [0.5, 1, 2.0, 4.5, np.float32(2.0), np.float64(1.5)])
    def test_a_usable_generate_dt_still_builds(self, value: Any) -> None:
        assert _config(generate_dt=value).controller_dt == pytest.approx(NUM_REGEN_FRAMES / 30.0 * float(value))

    def test_a_fractional_generate_dt_stays_first_class(self) -> None:
        # Unlike ``fps`` this is a continuous multiplier, so the two knobs take
        # different shared guards rather than one.
        assert _config(generate_dt=0.25).controller_dt == pytest.approx(NUM_REGEN_FRAMES / 30.0 * 0.25)

    def test_the_refusal_names_the_field_and_the_value(self) -> None:
        with pytest.raises(ValueError, match=r"MotionBricksConfig: generate_dt must be > 0, got 0\.0\."):
            _config(generate_dt=0.0)


class TestSpeedScaleDomain:
    """Both ``speed_scale`` components multiply the synthesised root velocity."""

    @pytest.mark.parametrize("index", [0, 1])
    @pytest.mark.parametrize("value", [0, 0.0, -1.0, *UNUSABLE_ANY_KNOB])
    def test_an_unusable_component_is_refused_naming_its_index(self, index: int, value: Any) -> None:
        pair: list[Any] = [1.0, 1.0]
        pair[index] = value
        with pytest.raises(ValueError, match=rf"speed_scale\[{index}\]"):
            _config(speed_scale=tuple(pair))

    def test_a_string_component_is_no_longer_laundered_into_a_float(self) -> None:
        # Pre-fix ``float()`` ran before the range test, so this was accepted and
        # STORED as ``(1.0, 2.0)`` - a config reporting floats the caller never
        # wrote.
        with pytest.raises(ValueError, match=r"MotionBricksConfig: speed_scale\[0\] must be > 0, got '1'\."):
            _config(speed_scale=("1", "2"))

    @pytest.mark.parametrize("value", [0.5, (1.0,), (1.0, 1.0, 1.0), (), np.float64(1.0)])
    def test_a_pair_of_the_wrong_arity_is_refused_as_an_arity_mistake(self, value: Any) -> None:
        # Including the scalar: pre-fix ``tuple(0.5)`` raised ``'float' object is
        # not iterable`` out of the arity check itself, past this message.
        with pytest.raises(ValueError, match=r"speed_scale must be a \(min, max\) pair"):
            _config(speed_scale=value)

    def test_a_two_character_string_is_refused_componentwise_not_as_an_arity_mistake(self) -> None:
        # ``"ab"`` DOES carry a component count of 2, so the arity rule has
        # nothing to say about it and the component domain is what refuses it.
        # Recorded because the two rules are easy to conflate.
        with pytest.raises(ValueError, match=r"MotionBricksConfig: speed_scale\[0\] must be > 0, got 'a'\."):
            _config(speed_scale="ab")

    def test_an_inverted_pair_is_still_refused_as_an_ordering_mistake(self) -> None:
        # Both components are individually usable, so this rule is the config's
        # own and cannot come from a shared scalar domain.
        with pytest.raises(ValueError, match=r"speed_scale must be 0 < min <= max, got \(2\.0, 1\.0\)"):
            _config(speed_scale=(2.0, 1.0))

    @pytest.mark.parametrize("value", [(1.0, 1.0), (0.8, 1.2), [0.8, 1.2], (2.0, 2.0), (np.float32(0.9), 1.1)])
    def test_a_usable_pair_still_builds_and_normalises_to_plain_floats(self, value: Any) -> None:
        scale = _config(speed_scale=value).speed_scale
        assert scale == pytest.approx((float(value[0]), float(value[1])))
        assert [type(component) for component in scale] == [float, float]


class TestWhyTheseDomainsAreWhatTheyAre:
    """The horizon each refused value would have produced, and where it goes.

    Without these the domain reads as tidying: it is not, because every value in
    the table reached the generator.
    """

    @pytest.mark.parametrize(
        ("fps", "generate_dt", "expected"),
        [
            (float("inf"), 2.0, 0.0),
            (True, 2.0, 16.0),
            (2.5, 2.0, 6.4),
            (30, float("inf"), float("inf")),
            (30, True, NUM_REGEN_FRAMES / 30.0),
        ],
    )
    def test_a_refused_knob_would_have_produced_this_horizon(self, fps: Any, generate_dt: Any, expected: float) -> None:
        assert _horizon(fps=fps, generate_dt=generate_dt) == pytest.approx(expected)

    @pytest.mark.parametrize("knob", ["fps", "generate_dt"])
    def test_a_non_finite_knob_would_have_poisoned_the_horizon(self, knob: str) -> None:
        args: dict[str, Any] = {"fps": 30, "generate_dt": 2.0, knob: float("nan")}
        assert np.isnan(_horizon(**args))

    def test_the_horizon_reaches_the_generator_on_every_call(self) -> None:
        # The reason the table above is a defect and not a curiosity: this is the
        # same value, unexamined by anything between the config and
        # ``full_navigation_agent.generate_new_frames``.
        agent = _RecordingAgent()
        config = _config(fps=60, generate_dt=0.5)
        policy = MotionBricksPolicy(config=config, motion_agent=agent, style="walk")
        policy.get_actions_sync({}, "")
        policy.get_actions_sync({}, "")
        assert agent.horizons == [pytest.approx(config.controller_dt)] * 2
        assert agent.horizons[0] == pytest.approx(NUM_REGEN_FRAMES / 60.0 * 0.5)


class TestResultDirPathDomain:
    """``result_dir`` names a location, so its domain is path-ness not truthiness.

    Measured on ``89410ad``, one ``MotionBricksConfig(result_dir=<value>)`` per
    row, no ``motionbricks`` install:

    ==================== ============================ ==========================
    value                pre-fix verdict              consequence
    ==================== ============================ ==========================
    ``123``              accepted, stored ``123``     ``Path(123)`` -> TypeError
    ``True``             accepted, stored ``True``    as above
    ``b"out"``           accepted, stored ``b'out'``  as above; pathlib refuses
                                                      bytes too
    ``["out"]``          accepted, stored ``['out']`` as above, AND
                                                      ``hash(config)`` raised
                                                      ``unhashable type: 'list'``
    ``0``                refused                      by the truthiness test, so
                                                      the message says "non-empty
                                                      path" about a number
    ``Path("out")``      accepted, stored as a        unequal to the identical
                         ``PosixPath``                 config built from ``"out"``
    ==================== ============================ ==========================

    The first four are the defect this closes; the last two are why the remedy is
    a normalising domain rather than an ``isinstance(str)`` gate, which would have
    refused the ``Path`` a caller is most likely to be holding.
    """

    @pytest.mark.parametrize("value", [123, 0, 3.5, True, False, None, b"out", ["out"], ("out",), {"dir": "out"}])
    def test_a_value_no_path_can_be_read_from_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"result_dir must be a str or os\.PathLike"):
            _config(result_dir=value)

    def test_the_refusal_names_the_value_its_type_and_where_it_would_have_failed(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                r"MotionBricksConfig\.result_dir must be a str or os\.PathLike path to the 'out/' "
                r"checkpoint tree, got 123 \(int\); it is read as Path\(result_dir\) when the generator "
                r"is built, which raises TypeError there rather than here\."
            ),
        ):
            _config(result_dir=123)

    def test_an_empty_result_dir_is_still_refused_as_an_empty_path(self) -> None:
        # Unchanged, and deliberately a separate message from the one above: an
        # empty string IS a path-shaped value, it just names nothing.
        with pytest.raises(ValueError, match=r"result_dir must be a non-empty path"):
            _config(result_dir="")

    @pytest.mark.parametrize("value", ["out", "~/out", "/srv/ckpt/out", "out/version_1"])
    def test_a_usable_path_string_still_builds_unchanged(self, value: str) -> None:
        assert _config(result_dir=value).result_dir == value

    def test_a_whitespace_only_path_stays_first_class(self) -> None:
        # A directory named with spaces is a legal path, so refusing it is an
        # allowlist decision rather than a path-ness one. Pinned so this guard
        # cannot creep into being one.
        assert _config(result_dir="  ").result_dir == "  "

    def test_a_path_is_accepted_and_normalised_to_the_str_the_field_declares(self) -> None:
        config = _config(result_dir=Path("~/out"))
        assert config.result_dir == "~/out"
        assert type(config.result_dir) is str

    def test_any_pathlike_is_accepted_not_only_pathlib(self) -> None:
        class _CheckpointTree:
            def __fspath__(self) -> str:
                return "out"

        assert _config(result_dir=_CheckpointTree()).result_dir == "out"

    def test_a_pathlike_whose_fspath_is_not_a_path_is_refused_not_raised_from_fspath(self) -> None:
        # ``os.PathLike`` is a duck-typed ABC, so this satisfies ``isinstance``
        # and ``os.fspath`` then raises. The refusal is the channel this check
        # answers on, so that raise must not escape past it.
        class _Broken:
            def __fspath__(self) -> str:
                return 5  # type: ignore[return-value]

        with pytest.raises(ValueError, match=r"result_dir must be a str or os\.PathLike"):
            _config(result_dir=_Broken())

    def test_two_configs_naming_the_same_tree_are_equal_and_hash_equal(self) -> None:
        # Pre-fix the ``Path`` was stored verbatim, so these compared unequal
        # while naming one directory - and this dataclass is frozen, so a caller
        # may legitimately put it in a set or a dict key.
        assert _config(result_dir=Path("out")) == _config(result_dir="out")
        assert hash(_config(result_dir=Path("out"))) == hash(_config(result_dir="out"))

    def test_an_empty_path_object_is_beyond_this_guards_reach(self) -> None:
        # Recorded as a limit, not an endorsement: ``Path("")`` IS ``Path(".")``
        # before any of this runs, so the empty string never arrives here as a
        # path object, and refusing the value that does arrive would mean
        # refusing ``Path(".")``.
        assert os.fspath(Path("")) == "."
        assert _config(result_dir=Path("")).result_dir == "."

    @pytest.mark.parametrize("value", [123, True, ["out"], b"out"])
    def test_a_refused_value_would_have_raised_out_of_the_generator_build(
        self, value: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Out of the real ``_build_agent``, applied to a namespace, for the same
        # reason ``_horizon`` is: a config carrying these values can no longer be
        # built. ``require_optional`` is the only thing between the config and
        # ``Path(result_dir)``, and it fails first only because the extra is
        # absent here - on a host with it installed the TypeError is what a
        # caller got.
        monkeypatch.setattr(
            "strands_robots.policies.motionbricks.policy.require_optional",
            lambda *a, **k: None,
        )
        with pytest.raises(TypeError, match=r"argument should be a str or an os\.PathLike"):
            MotionBricksPolicy._build_agent(SimpleNamespace(), SimpleNamespace(result_dir=value))  # type: ignore[arg-type]

    def test_the_policy_result_dir_shortcut_reaches_the_same_domain(self) -> None:
        # ``MotionBricksPolicy(result_dir=...)`` builds the config through
        # ``from_dict``, so the convenience shortcut is not a second door.
        shortcut: dict[str, Any] = {"result_dir": 123}
        with pytest.raises(ValueError, match=r"result_dir must be a str or os\.PathLike"):
            MotionBricksPolicy(motion_agent=_RecordingAgent(), **shortcut)


class TestTheConfigFilePathIsCovered:
    """A checkpoint's ``config.json`` reaches the same domain as the constructor."""

    @pytest.mark.parametrize(
        "bad",
        [
            {"fps": float("inf")},
            {"fps": 2.5},
            {"generate_dt": float("nan")},
            {"speed_scale": [float("nan"), 1.0]},
            {"speed_scale": ["1", "2"]},
            {"result_dir": 123},
            {"result_dir": None},
        ],
    )
    def test_from_dict_refuses_an_unusable_knob(self, bad: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            MotionBricksConfig.from_dict({"result_dir": "out", **bad})

    def test_from_file_refuses_a_result_dir_no_path_can_be_read_from(self, tmp_path: Path) -> None:
        # JSON carries a number natively, so a checkpoint's ``config.json`` can
        # hold one where a path belongs.
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"result_dir": 123}))
        with pytest.raises(ValueError, match=r"result_dir must be a str or os\.PathLike"):
            MotionBricksConfig.from_file(path)

    def test_from_file_refuses_an_unusable_knob(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        # ``Infinity`` is what ``json.dump`` writes for ``inf`` and what
        # ``json.loads`` reads back, so a JSON config can carry one.
        path.write_text(json.dumps({"result_dir": "out", "generate_dt": float("inf")}))
        with pytest.raises(ValueError, match=r"generate_dt must be > 0"):
            MotionBricksConfig.from_file(path)

    def test_from_file_still_loads_a_usable_config(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"result_dir": "out", "fps": 60, "generate_dt": 1.5, "speed_scale": [0.9, 1.1]}))
        config = MotionBricksConfig.from_file(path)
        assert (config.fps, config.generate_dt, config.speed_scale) == (60, 1.5, (0.9, 1.1))


class TestTheUpstreamDefaultsStayFirstClass:
    """The domain must not refuse the configuration the reference runner uses."""

    def test_the_defaults_build(self) -> None:
        config = _config()
        assert (config.fps, config.generate_dt, config.speed_scale) == (30, 2.0, (1.0, 1.0))
        assert config.controller_dt == pytest.approx(NUM_REGEN_FRAMES / 30.0 * 2.0)

    def test_the_upstream_perturbation_range_builds(self) -> None:
        assert _config(speed_scale=(0.7, 1.3)).speed_scale == pytest.approx((0.7, 1.3))


class TestNeighbouringConfigFieldsStayOutOfScope:
    """The enumeration fields are NOT part of the path domain above.

    A pin of current behaviour, not an endorsement of it: ``device=5`` and
    ``clips="g1"`` are both accepted today and fail inside ``torch`` or upstream
    ``motionbricks``. A non-empty-string check is not the rule they want - it
    would refuse the first and accept the second, which are the two rows a
    caller is most likely to write - because the useful domain here is
    membership, and where the member list lives is a decision rather than a
    copy. Tracked in #2010; per the premise-test guidance in ``AGENTS.md`` this
    boundary should be REPLACED rather than deleted when that lands.

    ``style`` is pinned here for the opposite reason: it is not silently
    honored downstream, so the config admitting it costs nothing.
    """

    @pytest.mark.parametrize("field", ["device", "clips", "exp"])
    def test_a_non_string_enumeration_field_is_still_accepted(self, field: str) -> None:
        assert getattr(_config(**{field: 5}), field) == 5

    def test_a_mistyped_member_name_is_still_accepted(self) -> None:
        # The row a type check would not have caught either: ``"G1"`` is the only
        # shipped clip set.
        assert _config(clips="g1").clips == "g1"

    def test_the_style_check_is_unchanged(self) -> None:
        # ``style`` admits an int index OR a str name, so it is neither a numeric
        # domain nor a path one and keeps its own local check.
        with pytest.raises(ValueError, match=r"style must be an int mode index or a str mode name"):
            _config(style=3.5)
        assert _config(style=True).style is True

    def test_a_bool_style_is_refused_where_the_mode_is_resolved(self) -> None:
        # Which is why the row above is not an open defect: the check that can
        # see the live clip list is the one placed to refuse it, and it names the
        # field and lists the modes.
        with pytest.raises(ValueError, match=r"style must be a mode index or name, not a bool"):
            resolve_mode(_config(style=True).style, ["idle", "walk"])


class TestOptionalPathFieldsShareThatDomain:
    """``skeleton_xml`` and ``scene_xml`` name locations, so path-ness is their domain too.

    Measured on ``4485c5e``, one ``MotionBricksConfig(skeleton_xml=<value>)`` per
    row, no ``motionbricks`` install, reading what reached
    ``full_navigation_agent(skeleton_xml=...)`` through the stub seam:

    ===================== ============================ ==========================
    value                 pre-fix verdict              reached the generator as
    ===================== ============================ ==========================
    ``None``              accepted                     the derived default -
                                                       correct, and documented
    ``"custom/g1.xml"``   accepted                     itself - correct
    ``Path("c/g1.xml")``  accepted, stored as a        a ``PosixPath``; unequal to
                          ``PosixPath``                the identical config built
                                                       from a ``str``
    ``""``                accepted                     the DERIVED DEFAULT - the
                                                       caller's field discarded
                                                       by ``or``, under success
    ``0``                 accepted                     as above
    ``123``               accepted, stored ``123``     ``123`` - fails inside
                                                       upstream, not here
    ``True``              accepted                     ``True`` - as above
    ``["custom/g1.xml"]`` accepted                     the list; AND
                                                       ``hash(config)`` raised
                                                       ``unhashable type: 'list'``
    ``b"g1.xml"``         accepted                     the bytes; pathlib refuses
                                                       bytes too
    ===================== ============================ ==========================

    The first row is the behaviour that must not change - ``None`` is how a
    caller asks the builder to derive the path - and it is why the two falsy rows
    are refusals rather than a widening of it: a build that answers "I used the
    default" to a caller who named a skeleton cannot be told from one that was
    asked to derive it.
    """

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    @pytest.mark.parametrize("value", [123, 0, 3.5, True, False, b"g1.xml", ["g1.xml"], ("g1.xml",), {"p": "g1.xml"}])
    def test_a_value_no_path_can_be_read_from_is_refused(self, field: str, value: Any) -> None:
        with pytest.raises(ValueError, match=rf"{field} must be a str or os\.PathLike"):
            _config(**{field: value})

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_the_refusal_names_the_field_the_value_its_type_and_where_it_would_have_failed(self, field: str) -> None:
        # The same four parts ``result_dir``'s refusal carries, so a caller who
        # has read one message can read all three.
        with pytest.raises(ValueError) as excinfo:
            _config(**{field: 123})
        message = str(excinfo.value)
        assert f"MotionBricksConfig.{field} must be a str or os.PathLike path to the G1 " in message
        assert "got 123 (int);" in message
        assert message.endswith("rather than here.")

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_omitting_the_field_stays_first_class(self, field: str) -> None:
        # The row the guard must not touch: ``None`` is the documented way to ask
        # the builder to derive this path, and it is unchanged by the fix.
        assert getattr(_config(**{field: None}), field) is None
        assert getattr(_config(), field) is None

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_an_empty_path_is_refused_and_the_message_names_the_way_to_derive_one(self, field: str) -> None:
        # Pre-fix this was accepted and then discarded for being falsy. It cannot
        # simply be read as ``None``: the two are different requests, and only one
        # of them says "I have no path".
        with pytest.raises(ValueError, match=rf"{field} must be a non-empty path to the G1 .* or None to derive it"):
            _config(**{field: ""})

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_a_usable_path_string_still_builds_unchanged(self, field: str) -> None:
        assert getattr(_config(**{field: "custom/g1.xml"}), field) == "custom/g1.xml"

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_a_path_is_accepted_and_normalised_to_the_str_the_field_declares(self, field: str) -> None:
        config = _config(**{field: Path("custom/g1.xml")})
        assert getattr(config, field) == "custom/g1.xml"
        assert type(getattr(config, field)) is str

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_two_configs_naming_the_same_file_are_equal_and_hash_equal(self, field: str) -> None:
        # The consequence that is not about the eventual file read: a caller
        # holding a ``Path`` built a config that compared unequal to the identical
        # config built from a ``str``.
        assert _config(**{field: Path("custom/g1.xml")}) == _config(**{field: "custom/g1.xml"})
        assert hash(_config(**{field: Path("custom/g1.xml")})) == hash(_config(**{field: "custom/g1.xml"}))

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_a_list_no_longer_leaves_this_frozen_dataclass_unhashable(self, field: str) -> None:
        # ``hash()`` on the pre-fix config raised ``unhashable type: 'list'`` -
        # a frozen dataclass that is not hashable.
        with pytest.raises(ValueError, match=rf"{field} must be a str or os\.PathLike"):
            hash(_config(**{field: ["g1.xml"]}))

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_any_pathlike_is_accepted_not_only_pathlib(self, field: str) -> None:
        class _SkeletonFile:
            def __fspath__(self) -> str:
                return "custom/g1.xml"

        assert getattr(_config(**{field: _SkeletonFile()}), field) == "custom/g1.xml"

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_a_pathlike_whose_fspath_is_not_a_path_is_refused_not_raised_from_fspath(self, field: str) -> None:
        # ``os.fspath`` raises TypeError on this; the refusal is still reported on
        # this field's own channel rather than escaping as that TypeError.
        class _Broken:
            def __fspath__(self) -> int:  # type: ignore[override]
                return 5

        with pytest.raises(ValueError, match=rf"{field} must be a str or os\.PathLike"):
            _config(**{field: _Broken()})

    @pytest.mark.parametrize("field", ["skeleton_xml", "scene_xml"])
    def test_the_domain_is_reached_through_from_dict_and_from_file(self, field: str, tmp_path: Path) -> None:
        # All three public surfaces funnel to ``__post_init__``, so one rule
        # covers the constructor, a dict and a JSON file alike.
        with pytest.raises(ValueError, match=rf"{field} must be a str or os\.PathLike"):
            MotionBricksConfig.from_dict({"result_dir": "out", field: 123})
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"result_dir": "out", field: 123}))
        with pytest.raises(ValueError, match=rf"{field} must be a str or os\.PathLike"):
            MotionBricksConfig.from_file(config_file)
