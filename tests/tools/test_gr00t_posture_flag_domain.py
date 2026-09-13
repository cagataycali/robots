"""``gr00t_inference`` refuses a posture flag it cannot read.

The tool tables its numeric options and its selector vocabularies per action,
and read its six posture flags - ``use_tensorrt``, ``http_server``,
``use_sim_policy_wrapper``, ``deterministic``, ``remove_volumes`` and ``force``
- raw. Every non-empty string is truthy, so the spellings a caller reaches for
when opting out selected the affirmative posture, and ``None`` or ``0`` took
the other branch without being a declared spelling of it. Measured on
``df1ea2a`` against stubbed docker and service layers:

* ``remove_volumes="false"`` under ``lifecycle="teardown"`` reached
  ``_remove_container`` as the string, which appends ``-v`` to ``docker rm``
  for it - the opt-out discarded the checkpoints ``False`` is documented to
  preserve;
* ``force="false"`` on ``build_image`` reached ``_build_image`` as the string;
* ``http_server="false"`` on ``start`` moved the port from 5555 to 8000;
* ``deterministic="false"`` on ``n1.6`` was refused as ``deterministic=True
  requires protocol='n1.7'`` - a refusal that branches on the flag inherits
  the inversion;
* ``use_tensorrt="false"`` switched the three dtype rows of the enumerable
  guard *on*, and ``use_tensorrt=0`` switched them *off*, carrying an
  unparseable dtype into the detached argv under ``status="error"`` from the
  orchestration rather than from the guard.

These tests pin that each flag an action consumes is refused unless it is a
boolean, before any handler runs; that the protocol gate no longer branches on
an unread value; that the dtype rows cannot be gated by a value that is not a
declared spelling of *off*; that an action consuming none of the six refuses
none of them; and that a flag added to the signature cannot skip the roster.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

# The module object rather than its members: the tests below replace the
# handlers each action dispatches to and read the roster the guard consults.
import strands_robots.tools.gr00t_inference as gi
from strands_robots.utils import boolean_flag_error

# One value per rejection reason of the shared posture domain: the two spellings
# of *off* that read as *on*, the integers that pass as a silent posture, and the
# two values that take a branch without being a declared spelling of it.
BAD_POSTURES: tuple[Any, ...] = ("false", "no", 0, 1, None, [])

# The flag each action is actually handed, which is what may be refused.
ACTION_FLAGS = tuple((key, flag) for key, flags in gi._ACTION_POSTURE_FLAGS.items() for flag in flags)

# Every handler an action dispatches to after the boundary guards. Each is
# replaced by a recorder so a refusal that arrives *after* dispatch is visible
# as a recorded call rather than as a docker invocation.
HANDLERS = (
    "_build_image",
    "_download_checkpoint",
    "_start_container",
    "_start_service",
    "_remove_container",
    "_stop_service",
)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    """Replace every post-guard handler with a recorder and make docker fatal.

    ``_lifecycle`` is left real so the ``teardown`` and ``full`` phases reach
    the recorded ``_remove_container`` / ``_build_image`` beneath it, which is
    where the pre-fix flag values were read.
    """
    calls: dict[str, list[dict[str, Any]]] = {name: [] for name in HANDLERS}

    def _recorder(name: str) -> Any:
        def _record(*args: Any, **kwargs: Any) -> dict[str, Any]:
            calls[name].append(kwargs | ({"args": args} if args else {}))
            return {"status": "success", "message": f"{name} recorded", "skipped": False}

        return _record

    for name in HANDLERS:
        monkeypatch.setattr(gi, name, _recorder(name))

    def _no_subprocess(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("a refused call reached subprocess.run")

    monkeypatch.setattr(gi.subprocess, "run", _no_subprocess)
    monkeypatch.setattr(gi, "_is_service_running", lambda _port: False)
    monkeypatch.setattr(gi, "_resolve_build_source", lambda: ("https://example.invalid/repo", "tag"))
    monkeypatch.setattr(gi, "_find_gr00t_containers", lambda: {"status": "success", "containers": []})
    monkeypatch.setattr(gi, "_list_running_services", lambda: {"status": "success", "services": []})
    monkeypatch.setattr(gi.time, "sleep", lambda _s: None)
    return calls


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool with agent-shaped values.

    The six flags are annotated ``bool`` and every value under test here is
    deliberately outside that annotation, so one ``**kwargs: Any`` helper
    states that once rather than scattering a suppression over every call.
    """
    return gi.gr00t_inference(**kwargs)


def _message(result: dict[str, Any]) -> str:
    return str(result.get("message", ""))


def _action_kwargs(key: str) -> dict[str, Any]:
    """The smallest set of options that carries ``key`` past the other guards.

    ``key`` is a roster key, so ``lifecycle:<phase>`` selects the phase. The
    protocol is ``n1.7`` so ``use_sim_policy_wrapper`` is a read flag rather
    than an inert one; the legacy-protocol cells set their own.
    """
    action, _, phase = key.partition(":")
    kwargs: dict[str, Any] = {"action": action, "protocol": "n1.7"}
    if phase:
        kwargs["lifecycle"] = phase
    if action in ("start", "restart"):
        kwargs["checkpoint_path"] = "/data/checkpoints/m"
    if action == "download_checkpoint" or phase == "full":
        kwargs["hf_repo"] = "nvidia/model"
    return kwargs


def _total_calls(recorded: dict[str, list[dict[str, Any]]]) -> int:
    return sum(len(calls) for calls in recorded.values())


class TestAPostureTheToolCannotReadIsRefused:
    """Every flag an action consumes is held to the shared boolean domain."""

    @pytest.mark.parametrize(("key", "flag"), ACTION_FLAGS)
    @pytest.mark.parametrize("value", BAD_POSTURES)
    def test_a_non_boolean_posture_is_refused_before_any_handler_runs(
        self, recorded: dict[str, list[dict[str, Any]]], key: str, flag: str, value: Any
    ) -> None:
        result = _call(**_action_kwargs(key) | {flag: value})

        action = key.partition(":")[0]
        assert result["status"] == "error"
        assert _message(result).startswith(f"gr00t_inference: {action}: {flag} must be a boolean")
        assert _total_calls(recorded) == 0, "a handler ran on a posture the tool cannot read"

    @pytest.mark.parametrize("value", BAD_POSTURES + (True, False, np.True_, np.False_))
    def test_the_posture_domain_matches_the_shared_helper(
        self, recorded: dict[str, list[dict[str, Any]]], value: Any
    ) -> None:
        refused = _call(**_action_kwargs("lifecycle:teardown") | {"remove_volumes": value})["status"] == "error"

        assert refused == (boolean_flag_error(value, "remove_volumes", "lifecycle") is not None)

    @pytest.mark.parametrize("action", ["status", "stop", "list", "find_containers"])
    def test_an_action_that_consumes_no_posture_refuses_none_of_them(
        self, recorded: dict[str, list[dict[str, Any]]], action: str
    ) -> None:
        """The rule the numeric table already follows: an unread option is not refused."""
        result = _call(
            action=action,
            use_tensorrt="false",
            http_server="false",
            use_sim_policy_wrapper="false",
            deterministic="false",
            remove_volumes="false",
            force="false",
        )

        assert "must be a boolean" not in _message(result)

    @pytest.mark.parametrize("protocol", ["n1.5", "n1.6"])
    def test_the_legacy_protocols_ignore_the_sim_wrapper_so_it_is_not_refused(
        self, recorded: dict[str, list[dict[str, Any]]], protocol: str
    ) -> None:
        """``--use-sim-policy-wrapper`` is emitted for N1.7 only, as ``denoising_steps`` is for the rest."""
        result = _call(
            action="start", checkpoint_path="/data/checkpoints/m", protocol=protocol, use_sim_policy_wrapper="false"
        )

        assert "use_sim_policy_wrapper must be" not in _message(result)
        assert len(recorded["_start_service"]) == 1


class TestTheRefusalPrecedesTheEffectTheTruthyValueHad:
    """The postures measured on the pre-fix tree, each now not taken."""

    def test_an_unreadable_volume_posture_removes_no_container(self, recorded: dict[str, list[dict[str, Any]]]) -> None:
        """Pre-fix the string reached ``docker rm`` as a truthy ``-v``."""
        result = _call(**_action_kwargs("lifecycle:teardown") | {"remove_volumes": "false"})

        assert result["status"] == "error"
        assert recorded["_remove_container"] == []

    def test_an_unreadable_force_posture_builds_no_image(self, recorded: dict[str, list[dict[str, Any]]]) -> None:
        result = _call(**_action_kwargs("build_image") | {"force": "false"})

        assert result["status"] == "error"
        assert recorded["_build_image"] == []

    def test_an_unreadable_transport_posture_starts_no_service_and_moves_no_port(
        self, recorded: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Pre-fix ``http_server="false"`` started the REST server on 8000."""
        result = _call(**_action_kwargs("start") | {"http_server": "false"})

        assert result["status"] == "error"
        assert recorded["_start_service"] == []


class TestTheProtocolGateNoLongerInheritsTheInversion:
    """The ``deterministic`` / ``n1.7`` gate branches on the flag, so it is graded after the domain."""

    @pytest.mark.parametrize("protocol", ["n1.5", "n1.6"])
    def test_a_truthy_spelling_of_off_is_refused_as_the_flag_and_not_as_the_wrapper(
        self, recorded: dict[str, list[dict[str, Any]]], protocol: str
    ) -> None:
        """Pre-fix the message described ``deterministic=True`` to a caller who spelled the opposite."""
        result = _call(action="start", checkpoint_path="/data/checkpoints/m", protocol=protocol, deterministic="false")

        assert result["status"] == "error"
        assert _message(result).startswith("gr00t_inference: start: deterministic must be a boolean")
        assert "requires protocol" not in _message(result)

    @pytest.mark.parametrize("protocol", ["n1.5", "n1.6"])
    def test_a_boolean_true_on_a_legacy_protocol_is_still_refused_by_the_gate(
        self, recorded: dict[str, list[dict[str, Any]]], protocol: str
    ) -> None:
        """Control: the gate keeps its job once the value it branches on is a boolean."""
        result = _call(action="start", checkpoint_path="/data/checkpoints/m", protocol=protocol, deterministic=True)

        assert result["status"] == "error"
        assert "requires protocol='n1.7'" in _message(result)
        assert recorded["_start_service"] == []

    @pytest.mark.parametrize("action", ["status", "stop", "list", "find_containers"])
    def test_an_action_that_never_reads_the_flag_is_not_gated_on_it(
        self, recorded: dict[str, list[dict[str, Any]]], action: str
    ) -> None:
        """The gate is scoped to the roster, so ``status`` is not refused for a wrapper it never mounts."""
        result = _call(action=action, protocol="n1.6", deterministic=True)

        assert "requires protocol" not in _message(result)


class TestTheGateCannotSwitchTheDtypeRows:
    """The three dtype rows are read under ``use_tensorrt``, so the gate is graded first."""

    def test_a_falsy_gate_no_longer_discards_the_dtype_rows(self, recorded: dict[str, list[dict[str, Any]]]) -> None:
        """Pre-fix ``use_tensorrt=0`` carried an unparseable dtype past the guard into the argv."""
        result = _call(**_action_kwargs("start") | {"use_tensorrt": 0, "vit_dtype": "fp8!"})

        assert result["status"] == "error"
        assert "use_tensorrt must be a boolean" in _message(result)
        assert recorded["_start_service"] == []

    def test_the_refusal_names_the_flag_rather_than_the_dtype_it_gates(
        self, recorded: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Placement pin: the enumerable guard would blame ``vit_dtype`` instead."""
        result = _call(**_action_kwargs("start") | {"use_tensorrt": "false", "vit_dtype": "fp8!"})

        assert _message(result).startswith("gr00t_inference: start: use_tensorrt must be a boolean")

    def test_a_boolean_gate_still_decides_whether_the_dtype_rows_are_read(
        self, recorded: dict[str, list[dict[str, Any]]]
    ) -> None:
        """The rows stay gated: a dtype the server never reads stands, one it reads is refused."""
        off = _call(**_action_kwargs("start") | {"use_tensorrt": False, "vit_dtype": "fp8!"})
        on = _call(**_action_kwargs("start") | {"use_tensorrt": True, "vit_dtype": "fp8!"})

        assert "vit_dtype must be" not in _message(off)
        assert len(recorded["_start_service"]) == 1
        assert on["status"] == "error"
        assert "vit_dtype must be a lowercase selector token" in _message(on)


class TestAnHonestPostureStillSelectsBothPaths:
    """The refusal must not cost a caller who supplies the declared domain."""

    @pytest.mark.parametrize(
        "value", [True, False, np.True_, np.False_], ids=["true", "false", "numpy-true", "numpy-false"]
    )
    def test_a_boolean_volume_posture_reaches_the_removal_unchanged(
        self, recorded: dict[str, list[dict[str, Any]]], value: Any
    ) -> None:
        result = _call(**_action_kwargs("lifecycle:teardown") | {"remove_volumes": value})

        assert result["status"] == "success"
        assert len(recorded["_remove_container"]) == 1
        assert recorded["_remove_container"][0]["remove_volumes"] is value

    @pytest.mark.parametrize(("http_server", "port"), [(True, 8000), (False, 5555)], ids=["http", "zmq"])
    def test_a_boolean_transport_posture_still_selects_its_port(
        self, recorded: dict[str, list[dict[str, Any]]], http_server: bool, port: int
    ) -> None:
        result = _call(**_action_kwargs("start") | {"http_server": http_server})

        assert result["status"] == "success"
        assert recorded["_start_service"][0]["port"] == port
        assert recorded["_start_service"][0]["http_server"] is http_server


class TestThePostureRosterFollowsTheSignature:
    """Structural pins: a flag cannot be added to the tool and skip the roster."""

    def test_every_boolean_parameter_of_the_tool_is_on_the_roster(self) -> None:
        parameters = inspect.signature(gi.gr00t_inference).parameters
        declared = {name for name, param in parameters.items() if param.annotation in (bool, "bool")}
        rostered = {flag for flags in gi._ACTION_POSTURE_FLAGS.values() for flag in flags}

        assert declared == rostered, f"posture flag read without the domain: {declared ^ rostered}"

    def test_every_action_with_gated_dtype_rows_rosters_the_gate(self) -> None:
        """Wherever the enumerable table reads the dtypes under ``use_tensorrt``, the gate is checked first."""
        for key in gi._ACTION_ENUMERABLE_OPTIONS:
            assert "use_tensorrt" in gi._ACTION_POSTURE_FLAGS.get(key, ()), key

    def test_every_roster_key_is_an_action_or_a_lifecycle_phase(self) -> None:
        """A key the dispatcher never produces would be a row nothing consults."""
        dispatched = {"build_image", "download_checkpoint", "start_container", "start", "restart"}
        phases = {"lifecycle:full", "lifecycle:teardown"}

        assert set(gi._ACTION_POSTURE_FLAGS) == dispatched | phases
