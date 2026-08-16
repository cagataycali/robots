"""``build_lerobot_command`` must refuse a numeric knob the lerobot CLI cannot honor.

Every numeric option this tool accepts is interpolated with ``str()`` into the
command line of a subprocess launched with ``start_new_session=True``. That
detached process is not a channel the call can read a failure back from: the
session starts, ``status="success"`` is returned with a pid, and a value the
lerobot CLI cannot parse surfaces minutes later in the session's log file. A
value the CLI *can* parse but should never have been given is worse still,
because nothing reports it at any point:

* ``dataset_fps=0`` put ``--dataset.fps 0`` on a ``lerobot-record`` argv;
* ``dataset_num_episodes=0`` asked for a recording of no episodes;
* ``replay_episode=-1`` put ``--dataset.episode -1`` on a replay;
* ``dataset_fps=nan`` / ``inf`` / ``None`` / ``[30]`` reached the argv as the
  literals ``nan``, ``inf``, ``None`` and ``[30]``.

Two knobs were read for truthiness rather than presence, which made ``0`` mean
the *opposite* of what it says: ``teleop_time_s=0`` ("stop at once") emitted no
budget at all, leaving an unbounded teleop session, and a replay with
``dataset_fps=0`` dropped the rate flag and took lerobot's own default instead
of the caller's.

The knobs are checked against the shared scalar domains
(:mod:`strands_robots.utils`) that every other recording surface in the tree
already applies, and only for the knobs the requested mode actually emits -
refusing a value a mode never puts on the argv would be a false rejection.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod

build_lerobot_command = tele_mod.build_lerobot_command
lerobot_teleoperate = tele_mod.lerobot_teleoperate


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Keep the module-level session store inside the test's temp dir."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(tele_mod, "SESSION_DIR", session_dir)
    return session_dir


# A value in none of the accepted domains: no run can be given it, and the CLI
# either rejects it minutes later or silently reads something else.
UNUSABLE = [
    pytest.param(0, id="zero"),
    pytest.param(-5, id="negative"),
    pytest.param(2.7, id="fractional"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(True, id="bool"),
    pytest.param("30", id="numeric-string"),
    pytest.param(None, id="none"),
    pytest.param([30], id="list"),
]

# Whole-number knobs accept an integral real, so a count read from a config or
# promoted by NumPy arithmetic is still honored.
INTEGRAL = [
    pytest.param(30, id="int"),
    pytest.param(30.0, id="integral-float"),
    pytest.param(np.int64(30), id="np-int64"),
]


def _record(**overrides: Any) -> list[str]:
    """A ``lerobot-record`` argv (``start`` + a dataset repo id)."""
    kwargs: dict[str, Any] = {
        "action": "start",
        "robot_type": "so101_follower",
        "robot_port": "/dev/ttyACM1",
        "teleop_type": "so101_leader",
        "teleop_port": "/dev/ttyACM0",
        "dataset_repo_id": "user/pick",
        "dataset_single_task": "pick the cube",
    }
    kwargs.update(overrides)
    return build_lerobot_command(**kwargs)


def _teleop(**overrides: Any) -> list[str]:
    """A ``lerobot-teleoperate`` argv (``start`` with no dataset)."""
    kwargs: dict[str, Any] = {
        "action": "start",
        "robot_type": "so101_follower",
        "robot_port": "/dev/ttyACM1",
        "teleop_type": "so101_leader",
        "teleop_port": "/dev/ttyACM0",
    }
    kwargs.update(overrides)
    return build_lerobot_command(**kwargs)


def _replay(**overrides: Any) -> list[str]:
    """A ``lerobot-replay`` argv."""
    kwargs: dict[str, Any] = {
        "action": "replay",
        "robot_type": "so101_follower",
        "robot_port": "/dev/ttyACM1",
        "dataset_repo_id": "user/pick",
    }
    kwargs.update(overrides)
    return build_lerobot_command(**kwargs)


def _flag(argv: list[str], flag: str) -> str | None:
    """The token following ``flag``, or ``None`` when the flag is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


class TestARecordingRateNoRunCanHonorIsRefused:
    """The rate the dataset is written at is the knob with the widest blast radius."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_record_refuses_an_unusable_dataset_fps(self, value: Any) -> None:
        with pytest.raises(ValueError, match="dataset_fps"):
            _record(dataset_fps=value)

    @pytest.mark.parametrize("value", INTEGRAL)
    def test_record_emits_an_integral_rate_as_a_whole_number(self, value: Any) -> None:
        """lerobot declares ``DatasetRecordConfig.fps`` an ``int``.

        A ``30.0`` read from a config is a usable rate, so it is accepted - but
        the argv must carry ``30``, not ``30.0``, because the CLI parses that
        token into an ``int`` field. The coercion is what makes accepting the
        integral float honest rather than a deferred failure.
        """
        assert _flag(_record(dataset_fps=value), "--dataset.fps") == "30"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_dagger_refuses_an_unusable_dataset_fps(self, value: Any) -> None:
        """The DAgger corrections it appends land in the same dataset."""
        with pytest.raises(ValueError, match="dataset_fps"):
            build_lerobot_command(
                action="dagger",
                robot_type="so101_follower",
                robot_port="/dev/ttyACM1",
                dataset_repo_id="user/pick",
                policy_path="lerobot/act_so101",
                dataset_fps=value,
            )


class TestZeroIsHonoredWhereItIsARealSettingAndRefusedWhereItIsNot:
    """The floor differs per knob, and both directions are pinned here.

    Making every knob strictly positive would reject two real requests; making
    them all non-negative would accept three that no run can satisfy. Neither
    single floor is correct, which is why the domains are chosen per knob.
    """

    def test_no_operator_pause_between_episodes_is_accepted(self) -> None:
        assert _flag(_record(dataset_reset_time_s=0), "--dataset.reset_time_s") == "0"

    def test_the_first_episode_is_accepted_for_replay(self) -> None:
        assert _flag(_replay(replay_episode=0), "--dataset.episode") == "0"

    @pytest.mark.parametrize(
        "knob",
        ["dataset_fps", "dataset_num_episodes", "dataset_episode_time_s"],
    )
    def test_a_zero_that_no_run_can_satisfy_is_refused(self, knob: str) -> None:
        """A zero rate, a zero-episode recording and a zero-length episode."""
        with pytest.raises(ValueError, match=knob):
            _record(**{knob: 0})

    @pytest.mark.parametrize("knob", ["dataset_reset_time_s", "replay_episode"])
    def test_the_non_negative_knobs_still_refuse_a_negative(self, knob: str) -> None:
        builder = _replay if knob == "replay_episode" else _record
        with pytest.raises(ValueError, match=knob):
            builder(**{knob: -1})


class TestATruthinessReadNoLongerInvertsAZero:
    """``0`` must not be read as "unset" for a knob whose zero is meaningful."""

    def test_a_zero_session_budget_is_refused_rather_than_dropped(self) -> None:
        """Pre-fix this emitted no ``--teleop_time_s`` at all.

        The one value that means "stop at once" produced a session with no time
        limit - the opposite of the request - and reported success.
        """
        with pytest.raises(ValueError, match="teleop_time_s"):
            _teleop(teleop_time_s=0)

    def test_an_omitted_session_budget_still_means_no_limit(self) -> None:
        """``None`` is the documented default and stays a supplied value."""
        assert _flag(_teleop(teleop_time_s=None), "--teleop_time_s") is None

    def test_a_usable_session_budget_is_emitted_and_may_be_fractional(self) -> None:
        """lerobot declares ``TeleoperateConfig.teleop_time_s`` a ``float``."""
        assert _flag(_teleop(teleop_time_s=12.5), "--teleop_time_s") == "12.5"

    def test_replay_always_emits_the_rate_it_was_given(self) -> None:
        """Pre-fix a falsy rate dropped the flag and took lerobot's default."""
        assert _flag(_replay(dataset_fps=25), "--dataset.fps") == "25"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_replay_refuses_an_unusable_rate_instead_of_dropping_it(self, value: Any) -> None:
        with pytest.raises(ValueError, match="dataset_fps"):
            _replay(dataset_fps=value)


class TestOnlyTheKnobsAModeEmitsAreChecked:
    """A caller must never be refused for a value the requested mode ignores."""

    def test_teleoperate_ignores_the_dataset_rate(self) -> None:
        """No ``--dataset.*`` flag is emitted, so the rate is not read."""
        argv = _teleop(dataset_fps=0)
        assert _flag(argv, "--dataset.fps") is None
        assert _flag(argv, "--fps") == "60"

    def test_record_ignores_the_teleop_session_budget(self) -> None:
        argv = _record(teleop_time_s=0)
        assert _flag(argv, "--teleop_time_s") is None
        assert _flag(argv, "--dataset.fps") == "30"

    def test_record_ignores_the_replay_episode_index(self) -> None:
        assert _flag(_record(replay_episode=-1), "--dataset.episode") is None

    def test_replay_ignores_the_episode_count_and_time_budgets(self) -> None:
        argv = _replay(dataset_num_episodes=0, dataset_episode_time_s=-1, dataset_reset_time_s=-1)
        assert _flag(argv, "--dataset.num_episodes") is None
        assert _flag(argv, "--dataset.episode_time_s") is None


class TestTheRefusalPrecedesTheDetachedProcess:
    """Nothing may be launched or persisted for a call that cannot be honored."""

    def test_the_tool_reports_the_refusal_without_starting_a_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _never(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("subprocess.Popen must not be reached for a refused call")

        monkeypatch.setattr(tele_mod.subprocess, "Popen", _never)
        result = lerobot_teleoperate(
            action="start",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM1",
            teleop_type="so101_leader",
            teleop_port="/dev/ttyACM0",
            dataset_repo_id="user/pick",
            dataset_fps=0,
            session_name="refused",
        )
        assert result["status"] == "error"
        text = "\n".join(item.get("text", "") for item in result["content"] if "text" in item)
        assert "dataset_fps" in text
        assert tele_mod.SessionManager().get_session("refused") is None


class TestTheOptionTablesCannotDriftApart:
    """A knob added to a mode must have a domain, and vice versa."""

    def test_every_knob_a_mode_emits_has_a_declared_domain(self) -> None:
        domains = {name for name, _ in tele_mod._OPTION_DOMAINS}
        for mode, knobs in tele_mod._MODE_NUMERIC_OPTIONS.items():
            missing = sorted(set(knobs) - domains)
            assert not missing, f"mode {mode!r} emits {missing} with no domain in _OPTION_DOMAINS"

    def test_every_declared_domain_is_emitted_by_some_mode(self) -> None:
        """A domain no mode reads is dead weight that will rot."""
        emitted = {knob for knobs in tele_mod._MODE_NUMERIC_OPTIONS.values() for knob in knobs}
        orphans = sorted({name for name, _ in tele_mod._OPTION_DOMAINS} - emitted)
        assert not orphans, f"_OPTION_DOMAINS entries no mode emits: {orphans}"

    def test_every_declared_domain_names_a_real_parameter(self) -> None:
        params = set(inspect.signature(build_lerobot_command).parameters)
        unknown = sorted({name for name, _ in tele_mod._OPTION_DOMAINS} - params)
        assert not unknown, f"_OPTION_DOMAINS names non-parameters: {unknown}"

    def test_the_modes_are_exactly_the_ones_the_builder_dispatches(self) -> None:
        assert set(tele_mod._MODE_NUMERIC_OPTIONS) == {"replay", "record", "teleoperate", "dagger"}

    def test_an_optional_knob_is_one_the_table_actually_declares(self) -> None:
        domains = {name for name, _ in tele_mod._OPTION_DOMAINS}
        assert tele_mod._OPTIONAL_OPTIONS <= domains


class TestNoWholeNumberFlagCarriesANonIntegerToken:
    """Each accepted whole-number knob reaches the CLI as an ``int`` literal."""

    @pytest.mark.parametrize(
        ("flag", "knob", "builder"),
        [
            ("--dataset.fps", "dataset_fps", _record),
            ("--dataset.num_episodes", "dataset_num_episodes", _record),
            ("--dataset.episode_time_s", "dataset_episode_time_s", _record),
            ("--dataset.reset_time_s", "dataset_reset_time_s", _record),
            ("--fps", "fps", _teleop),
            ("--dataset.episode", "replay_episode", _replay),
        ],
    )
    def test_an_integral_float_is_emitted_without_a_decimal_point(self, flag, knob, builder) -> None:
        token = _flag(builder(**{knob: 7.0}), flag)
        assert token == "7", f"{flag} carried {token!r}, which lerobot parses into an int field"
