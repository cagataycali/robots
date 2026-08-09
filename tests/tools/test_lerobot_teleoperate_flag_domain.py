"""``build_lerobot_command`` must refuse a boolean mode flag it can only misread.

Every flag this builder emits selects a *posture*, not a magnitude: ``dataset_video``
picks the literal ``"true"`` or ``"false"`` on the argv, while ``record_resume``
decides whether ``--resume true`` appears at all. Both were read by truthiness,
and every non-empty string is truthy, so the words an operator reaches for when
opting out selected the opposite posture from the one they read as. Measured on
``3ce3da7``:

* ``dataset_push_to_hub="false"`` emitted ``--dataset.push_to_hub true``, so a
  detached, unattended recording uploaded its dataset to the Hub;
* ``record_resume="false"`` emitted ``--resume true``, appending into an existing
  dataset - and preserving its already-stamped repo_id - instead of creating the
  fresh one that was asked for;
* ``dagger_record_autonomous="off"`` emitted ``--strategy.record_autonomous
  true``, recording autonomous rollout episodes into a corrections dataset;
* ``display_data="false"`` emitted ``--display_data true``.

``None`` and ``[]`` took the other branch just as silently, without ever being a
declared spelling of it. None of these is reported anywhere: the argv goes to a
subprocess launched with ``start_new_session=True``, the tool returns
``status="success"`` with a pid, and the CLI parses every one of these argvs
without complaint - it is simply told the opposite posture.

The flags are checked against the shared
:func:`~strands_robots.utils.boolean_flag_error` domain, and only for the flags
the requested mode actually emits: refusing a flag a mode never puts on the argv
would be a false rejection, which is the same scoping rule the numeric knobs use
(``tests/tools/test_lerobot_teleoperate_numeric_domain.py``).
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod
from strands_robots.utils import boolean_flag_error

build_lerobot_command = tele_mod.build_lerobot_command
lerobot_teleoperate = tele_mod.lerobot_teleoperate


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Keep the module-level session store inside the test's temp dir."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(tele_mod, "SESSION_DIR", session_dir)
    return session_dir


@pytest.fixture
def _rollout_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present the lerobot rollout module so the ``dagger`` preflight passes.

    ``dagger`` needs lerobot>=0.6.0 installed to reach its argv at all; the flag
    refusal is deliberately placed *before* that preflight, so these tests pin
    the refusal on both sides of it (see
    :class:`TestTheRefusalPrecedesTheLerobotVersionPreflight`).
    """
    real_find_spec = tele_mod.importlib.util.find_spec

    def _find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "lerobot.scripts.lerobot_rollout":
            return object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(tele_mod.importlib.util, "find_spec", _find_spec)


# A value in no boolean's domain. The four strings are the spellings an operator
# reaches for when opting out, and each is truthy; ``nan`` and ``0.7`` are truthy
# numbers; ``None`` and ``[]`` are falsy values that are not a declared spelling
# of the negative posture either.
NOT_A_BOOLEAN = [
    pytest.param("false", id="str-false"),
    pytest.param("no", id="str-no"),
    pytest.param("off", id="str-off"),
    pytest.param("0", id="str-zero"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(0.7, id="fractional"),
    pytest.param(1, id="int-one"),
    pytest.param(0, id="int-zero"),
    pytest.param(None, id="none"),
    pytest.param([], id="empty-list"),
]

# Both python spellings plus the numpy booleans ``boolean_flag_error`` accepts,
# which arrive from an array-shaped config or a NumPy comparison.
A_BOOLEAN = [
    pytest.param(True, True, id="true"),
    pytest.param(False, False, id="false"),
    pytest.param(np.True_, True, id="np-true"),
    pytest.param(np.False_, False, id="np-false"),
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


def _dagger(**overrides: Any) -> list[str]:
    """A ``lerobot-rollout --strategy.type=dagger`` argv."""
    kwargs: dict[str, Any] = {
        "action": "dagger",
        "robot_type": "so101_follower",
        "robot_port": "/dev/ttyACM1",
        "teleop_type": "so101_leader",
        "teleop_port": "/dev/ttyACM0",
        "dataset_repo_id": "user/pick",
        "policy_path": "lerobot/act_so101",
    }
    kwargs.update(overrides)
    return build_lerobot_command(**kwargs)


def _token(argv: list[str], flag: str) -> str | None:
    """The token following ``flag``, or ``None`` when the flag is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


class TestAnUnattendedRecordingCannotBeTalkedIntoUploading:
    """``dataset_push_to_hub`` is the flag with the widest blast radius.

    A recording session is detached and unattended by design, and the Hub push
    happens at the end of it. A truthy spelling of off therefore published a
    dataset with no one watching, and the call that asked for the opposite had
    already returned ``status="success"``.
    """

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_record_refuses_a_non_boolean_push_to_hub(self, value: Any) -> None:
        with pytest.raises(ValueError, match="dataset_push_to_hub"):
            _record(dataset_push_to_hub=value)

    def test_the_opt_out_spelling_no_longer_selects_the_upload(self) -> None:
        """Pre-fix this emitted ``--dataset.push_to_hub true``."""
        with pytest.raises(ValueError, match="dataset_push_to_hub"):
            _record(dataset_push_to_hub="false")

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_dagger_refuses_a_non_boolean_push_to_hub(self, value: Any, _rollout_entry_point: None) -> None:
        """DAgger appends corrections to a dataset and pushes the same way."""
        with pytest.raises(ValueError, match="dataset_push_to_hub"):
            _dagger(dataset_push_to_hub=value)

    @pytest.mark.parametrize(("value", "expected"), A_BOOLEAN)
    def test_a_boolean_still_selects_the_posture_it_names(self, value: Any, expected: bool) -> None:
        token = _token(_record(dataset_push_to_hub=value), "--dataset.push_to_hub")
        assert token == ("true" if expected else "false")


class TestAFreshRecordingCannotBeTurnedIntoAnAppend:
    """``record_resume`` chooses between two datasets, not between two verbosities.

    Resume preserves an existing, already-stamped ``repo_id`` and appends to the
    data at the resolved root; a fresh record stamps a new one. A truthy spelling
    of off silently merged one operator's episodes into another's dataset.
    """

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_record_refuses_a_non_boolean_resume(self, value: Any) -> None:
        with pytest.raises(ValueError, match="record_resume"):
            _record(record_resume=value)

    def test_the_opt_out_spelling_no_longer_selects_the_append(self) -> None:
        """Pre-fix this emitted ``--resume true``."""
        with pytest.raises(ValueError, match="record_resume"):
            _record(record_resume="false")

    def test_a_true_resume_still_emits_the_flag(self) -> None:
        assert _token(_record(record_resume=True), "--resume") == "true"

    def test_a_false_resume_still_omits_the_flag(self) -> None:
        """Absence is how a fresh record is spelled; that is unchanged."""
        assert "--resume" not in _record(record_resume=False)

    def test_a_numpy_false_resume_omits_the_flag_too(self) -> None:
        """The check accepts a numpy boolean, so the emitter must handle one."""
        assert "--resume" not in _record(record_resume=np.False_)


class TestTheRemainingFlagsShareTheSameDomain:
    """``dataset_video``, ``display_data`` and ``dagger_record_autonomous``."""

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_record_refuses_a_non_boolean_video_setting(self, value: Any) -> None:
        with pytest.raises(ValueError, match="dataset_video"):
            _record(dataset_video=value)

    @pytest.mark.parametrize(("value", "expected"), A_BOOLEAN)
    def test_a_boolean_video_setting_reaches_the_argv(self, value: Any, expected: bool) -> None:
        assert _token(_record(dataset_video=value), "--dataset.video") == ("true" if expected else "false")

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_record_refuses_a_non_boolean_display_data(self, value: Any) -> None:
        with pytest.raises(ValueError, match="display_data"):
            _record(display_data=value)

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_teleoperate_refuses_a_non_boolean_display_data(self, value: Any) -> None:
        with pytest.raises(ValueError, match="display_data"):
            _teleop(display_data=value)

    def test_a_true_display_data_still_emits_the_flag(self) -> None:
        assert _token(_teleop(display_data=True), "--display_data") == "true"

    def test_a_false_display_data_still_omits_the_flag(self) -> None:
        assert "--display_data" not in _teleop(display_data=False)

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_dagger_refuses_a_non_boolean_record_autonomous(self, value: Any, _rollout_entry_point: None) -> None:
        with pytest.raises(ValueError, match="dagger_record_autonomous"):
            _dagger(dagger_record_autonomous=value)

    def test_a_true_record_autonomous_still_emits_the_flag(self, _rollout_entry_point: None) -> None:
        argv = _dagger(dagger_record_autonomous=True)
        assert _token(argv, "--strategy.record_autonomous") == "true"

    def test_a_false_record_autonomous_still_omits_the_flag(self, _rollout_entry_point: None) -> None:
        assert "--strategy.record_autonomous" not in _dagger(dagger_record_autonomous=False)


class TestOnlyTheFlagsAModeEmitsAreChecked:
    """A caller must never be refused for a flag the requested mode ignores.

    This is the over-reach control for the whole change: the same unusable value
    that is refused above must be accepted here, because no argv carries it.
    """

    @pytest.mark.parametrize(
        "flag",
        ["record_resume", "dataset_push_to_hub", "dataset_video", "display_data", "dagger_record_autonomous"],
    )
    def test_replay_refuses_no_flag_and_its_argv_is_unchanged(self, flag: str) -> None:
        """``lerobot-replay`` emits no boolean flag at all."""
        assert _replay(**{flag: "false"}) == _replay()

    @pytest.mark.parametrize(
        "flag",
        ["record_resume", "dataset_push_to_hub", "dataset_video", "dagger_record_autonomous"],
    )
    def test_teleoperate_reads_only_display_data(self, flag: str) -> None:
        """Plain teleoperation emits no ``--dataset.*`` flag and no strategy."""
        assert _teleop(**{flag: "false"}) == _teleop()

    def test_record_ignores_the_dagger_strategy_flag(self) -> None:
        assert _record(dagger_record_autonomous="false") == _record()

    def test_dagger_ignores_the_record_resume_flag(self, _rollout_entry_point: None) -> None:
        """``lerobot-rollout`` has no ``--resume``; the flag is never emitted."""
        assert _dagger(record_resume="false") == _dagger()


class TestPlaySoundsIsExcludedByConstruction:
    """No mode emits ``play_sounds``, so no mode may refuse a value for it.

    The parameter is declared, documented and forwarded, and then read by
    nothing - so giving it a domain here would be a false rejection for an option
    that has no effect either way. It is absent from the table for the same
    reason ``replay`` is, rather than by an exemption. Whether it should be
    emitted or removed is #2072; this class pins the state that issue describes,
    so it fails the day the answer lands and the exclusion stops being true.
    """

    def test_no_mode_emits_it(self) -> None:
        for name, tuple_ in tele_mod._MODE_FLAG_OPTIONS.items():
            assert "play_sounds" not in tuple_, f"mode {name!r} now emits play_sounds; give it a domain"

    @pytest.mark.parametrize("builder", [_record, _teleop, _replay])
    def test_the_argv_is_identical_either_way(self, builder: Any) -> None:
        assert builder(play_sounds=True) == builder(play_sounds=False)

    def test_the_dagger_argv_is_identical_either_way(self, _rollout_entry_point: None) -> None:
        assert _dagger(play_sounds=True) == _dagger(play_sounds=False)

    def test_no_argv_carries_a_sound_token(self) -> None:
        for argv in (_record(), _teleop(), _replay()):
            assert [token for token in argv if "sound" in token.lower()] == []

    def test_a_non_boolean_is_consequently_not_refused(self) -> None:
        """Not an oversight: nothing reads it, so nothing can misread it."""
        assert _record(play_sounds="false") == _record()


class TestTheRefusalPrecedesEverythingItWouldOtherwiseReach:
    """Nothing may be launched, persisted or preflighted for a refused call."""

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
            dataset_push_to_hub="false",
            session_name="refused-flag",
        )
        assert result["status"] == "error"
        text = "\n".join(item.get("text", "") for item in result["content"] if "text" in item)
        assert "dataset_push_to_hub" in text
        assert tele_mod.SessionManager().get_session("refused-flag") is None

    def test_dagger_refuses_the_flag_without_the_rollout_entry_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same caller mistake must report the same way on any lerobot.

        Placed before the version preflight, so an unusable flag is named rather
        than being masked by an upgrade hint on an older install.
        """
        monkeypatch.setattr(tele_mod.importlib.util, "find_spec", lambda *args, **kwargs: None)
        with pytest.raises(ValueError, match="dagger_record_autonomous"):
            _dagger(dagger_record_autonomous="false")

    def test_a_numeric_refusal_still_comes_first(self) -> None:
        """Both are refusals; the order is chosen rather than incidental.

        The numeric knobs are checked first so their messages - and the tests
        that pin them - are unchanged by this addition.
        """
        with pytest.raises(ValueError, match="dataset_fps"):
            _record(dataset_fps=0, dataset_push_to_hub="false")


class TestTheFlagTableCannotDriftFromTheBuilder:
    """The table is the record of what each mode emits; keep it measurable."""

    def test_every_flag_a_mode_emits_is_a_real_parameter(self) -> None:
        params = set(inspect.signature(build_lerobot_command).parameters)
        named = {flag for flags in tele_mod._MODE_FLAG_OPTIONS.values() for flag in flags}
        unknown = sorted(named - params)
        assert not unknown, f"_MODE_FLAG_OPTIONS names non-parameters: {unknown}"

    def test_every_flag_a_mode_emits_is_declared_a_bool(self) -> None:
        """A flag whose annotation is not ``bool`` belongs to another domain."""
        params = inspect.signature(build_lerobot_command).parameters
        named = {flag for flags in tele_mod._MODE_FLAG_OPTIONS.values() for flag in flags}
        # The module has no ``from __future__ import annotations``, so the
        # annotation is the ``bool`` type itself; accept the string spelling too
        # so adding that import does not silently disarm this check.
        adrift = sorted(flag for flag in named if params[flag].annotation not in (bool, "bool"))
        assert not adrift, f"_MODE_FLAG_OPTIONS names non-bool parameters: {adrift}"

    def test_every_mode_in_the_table_is_one_the_builder_dispatches(self) -> None:
        assert set(tele_mod._MODE_FLAG_OPTIONS) <= set(tele_mod._MODE_NUMERIC_OPTIONS)

    def test_replay_is_absent_because_it_emits_no_flag(self) -> None:
        """Absence is the assertion, so a flag added to replay fails here."""
        assert "replay" not in tele_mod._MODE_FLAG_OPTIONS
        assert "replay" in tele_mod._MODE_NUMERIC_OPTIONS

    def test_no_mode_names_a_flag_the_supplied_dict_cannot_answer(self) -> None:
        """Guards the ``supplied[param]`` lookup against a typo in the table."""
        every_flag_usable = dict.fromkeys(
            {flag for flags in tele_mod._MODE_FLAG_OPTIONS.values() for flag in flags}, True
        )
        for mode in tele_mod._MODE_FLAG_OPTIONS:
            assert tele_mod._flag_error(mode, every_flag_usable) is None

    def test_a_mode_absent_from_the_table_refuses_nothing(self) -> None:
        assert tele_mod._flag_error("replay", {}) is None

    def test_the_flags_are_reported_in_the_order_the_argv_emits_them(self) -> None:
        """Two unusable flags in one call must report deterministically."""
        with pytest.raises(ValueError, match="record_resume"):
            _record(record_resume="false", dataset_video="false")

    def test_the_flag_and_numeric_tables_name_disjoint_options(self) -> None:
        """A knob is a magnitude or a posture, never both."""
        numeric = {knob for knobs in tele_mod._MODE_NUMERIC_OPTIONS.values() for knob in knobs}
        flags = {flag for flags in tele_mod._MODE_FLAG_OPTIONS.values() for flag in flags}
        assert not numeric & flags


class TestTheSharedDomainIsTheOneApplied:
    """The refusal must be the shared one, not a local equivalent of it."""

    def test_the_message_is_the_shared_domains_message(self) -> None:
        expected = boolean_flag_error("false", "dataset_push_to_hub", "build_lerobot_command")
        assert expected is not None
        with pytest.raises(ValueError) as excinfo:
            _record(dataset_push_to_hub="false")
        assert str(excinfo.value) == expected

    def test_the_message_names_the_builder_as_the_context(self) -> None:
        with pytest.raises(ValueError, match="build_lerobot_command"):
            _record(dataset_video="false")

    def test_the_message_explains_why_it_is_not_parsed(self) -> None:
        """A caller who wrote ``"false"`` needs to know it was not read as off."""
        with pytest.raises(ValueError, match="checked rather than parsed"):
            _record(dataset_video="false")


class TestTheToolsOwnExecutionFlagsStayOutOfScope:
    """``background`` and ``auto_accept_calibration`` are not builder flags.

    Neither reaches an argv: they choose how ``lerobot_teleoperate`` runs the
    command it built - detached with a log file, and whether a newline is written
    to the child's stdin to accept a calibration prompt. They are parameters of
    the tool alone, have no per-mode table to be scoped by, and are read after
    the builder has returned. Pinned rather than assumed, so this boundary is
    measured; tracked separately.
    """

    @pytest.mark.parametrize("flag", ["background", "auto_accept_calibration"])
    def test_they_are_tool_parameters_and_not_builder_parameters(self, flag: str) -> None:
        assert flag in inspect.signature(lerobot_teleoperate).parameters
        assert flag not in inspect.signature(build_lerobot_command).parameters

    @pytest.mark.parametrize("flag", ["background", "auto_accept_calibration"])
    def test_the_builder_neither_reads_nor_refuses_them(self, flag: str) -> None:
        """They arrive in ``**kwargs`` and are ignored, as they were before."""
        assert _record(**{flag: "false"}) == _record()

    @pytest.mark.parametrize("flag", ["background", "auto_accept_calibration"])
    def test_they_are_not_in_the_flag_table(self, flag: str) -> None:
        named = {name for names in tele_mod._MODE_FLAG_OPTIONS.values() for name in names}
        assert flag not in named
