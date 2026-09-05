"""One security posture, one owner: the allowlist advisory must follow the transport.

An allowlist configured through ``DEVICE_CONNECT_RPC_ALLOW`` is only a
cryptographic authorization boundary under authenticated transport. Under
insecure transport the caller id is self-asserted, so
:func:`~strands_robots.device_connect._authz.is_authorized_caller` logs a
one-time advisory saying the enforcement is advisory. That advisory is the only
signal an operator gets, and it has to agree with the posture the transport is
actually running.

A ``DeviceRuntime`` resolves that posture from two sources with a documented
precedence - its own ``allow_insecure`` argument first, then
``DEVICE_CONNECT_ALLOW_INSECURE`` - which is the same precedence
:func:`~strands_robots.device_connect.resolve_allow_insecure` implements. The
advisory read the environment variable alone, so it answered only the
lower-precedence half of the question and disagreed with the transport on both
argument paths:

* ``allow_insecure=True`` with the variable unset is insecure and went
  **unwarned**, so a configured allowlist read as a boundary it was not. This is
  the shape the package's own guide documents for ``ReachyMiniDriver``, which
  builds its ``DeviceRuntime`` directly with ``allow_insecure=True`` and no
  environment variable at all.
* ``allow_insecure=False`` while the variable opts in is authenticated and was
  warned about anyway, and the message asserted the variable "is set" as the
  reason - true of the variable, not of the transport.

The pins that existed graded one source each. ``test_insecure_acl_logs_advisory_once``
and ``test_secure_acl_no_insecure_advisory`` both vary only the environment
variable, and ``test_allow_insecure_setting_domain`` grades the resolver in
isolation and never asks whether anything downstream honours its answer. So the
argument - the higher-precedence source - was absent from every cell that grades
the advisory, and the two rows here that vary only the environment are the
controls that hold either way.
"""

import asyncio
import logging
from typing import Any

import pytest

#: ``(label, allow_insecure argument, environment value, the resolved posture)``.
#: The two argument rows are where the advisory and the transport disagreed; the
#: two environment-only rows are the controls the existing pins already cover.
POSTURES: tuple[tuple[str, bool | None, str | None, bool], ...] = (
    ("the variable opts in, no argument", None, "true", True),
    ("the argument opts in, variable unset", True, None, True),
    ("the argument opts out, variable opts in", False, "true", False),
    ("neither opts in", None, None, False),
)


def _authz() -> Any:
    """The authorization module, imported inside the call as siblings do."""
    import strands_robots.device_connect._authz as az

    return az


class _Runtime:
    """A ``DeviceRuntime`` stand-in carrying only the resolved posture.

    ``DeviceRuntime`` resolves ``allow_insecure`` in its own ``__init__`` and
    hands itself to the driver through ``DeviceDriver.set_device``, so the
    attribute is all the advisory needs from it.
    """

    def __init__(self, allow_insecure: bool) -> None:
        self.allow_insecure = allow_insecure


def _advisories(caplog: pytest.LogCaptureFixture, **kwargs: Any) -> list[str]:
    """Ask ``is_authorized_caller`` once and return the advisories it logged."""
    az = _authz()
    az._warned_insecure_acl.clear()
    az._warned_permissive.clear()
    with caplog.at_level(logging.WARNING, logger="strands_robots.device_connect._authz"):
        caplog.clear()
        assert az.is_authorized_caller("ctrl", scope="rpc", **kwargs) is True
    return [r.getMessage() for r in caplog.records if "SELF-ASSERTED" in r.getMessage()]


@pytest.fixture(autouse=True)
def _allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """An allowlist is configured, which is what makes the advisory reachable."""
    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "ctrl")
    monkeypatch.delenv("DEVICE_CONNECT_ESTOP_ALLOW", raising=False)
    monkeypatch.delenv("DEVICE_CONNECT_ALLOW_INSECURE", raising=False)


class TestTheAdvisoryFollowsTheTransport:
    """The advisory fires exactly when the transport carrying the call is insecure."""

    @pytest.mark.parametrize(("label", "arg", "env", "insecure"), POSTURES, ids=[p[0] for p in POSTURES])
    def test_the_advisory_agrees_with_the_resolved_posture(
        self,
        label: str,
        arg: bool | None,
        env: str | None,
        insecure: bool,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        if env is not None:
            monkeypatch.setenv("DEVICE_CONNECT_ALLOW_INSECURE", env)
        # The runtime resolves the two sources with the argument outranking the
        # variable, so its ``allow_insecure`` is the posture in force.
        device = _Runtime(insecure)
        assert bool(_advisories(caplog, device=device)) is insecure, label

    def test_the_expected_postures_are_the_resolvers_own_answers(self) -> None:
        """The table above is measured against the resolver, not asserted by hand."""
        import strands_robots.device_connect as dc

        for label, arg, env, insecure in POSTURES:
            assert dc.resolve_allow_insecure(arg, env) is insecure, label

    def test_an_unattached_driver_still_falls_back_to_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With no runtime attached the variable is the only answer available."""
        monkeypatch.setenv("DEVICE_CONNECT_ALLOW_INSECURE", "yes")
        assert _advisories(caplog, device=None)
        monkeypatch.setenv("DEVICE_CONNECT_ALLOW_INSECURE", "no")
        assert not _advisories(caplog, device=None)

    def test_the_advisory_does_not_claim_the_variable_is_set(self, caplog: pytest.LogCaptureFixture) -> None:
        """Insecure by argument means the variable is unset, so the reason cannot cite it."""
        messages = _advisories(caplog, device=_Runtime(True))
        assert len(messages) == 1
        assert "DEVICE_CONNECT_ALLOW_INSECURE is set" not in messages[0]
        assert "allow_insecure" in messages[0]


class TestADriverCarriesItsOwnTransportToTheCheck:
    """The consequence, driven through a driver's RPC rather than the helper."""

    @pytest.mark.parametrize("insecure", [True, False])
    def test_an_rpc_reports_the_posture_of_the_runtime_it_is_attached_to(
        self, insecure: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

        az = _authz()
        az._warned_insecure_acl.clear()
        az._warned_permissive.clear()

        class _Sim:
            def __init__(self) -> None:
                self.steps = 0

            def step(self, n_steps: int) -> dict[str, Any]:
                self.steps += n_steps
                return {"status": "success", "content": [{"text": f"stepped {n_steps}"}]}

        sim = _Sim()
        driver = SimulationDeviceDriver(sim)
        # What ``DeviceRuntime.__init__`` does: hand the runtime to the driver.
        driver.set_device(_Runtime(insecure))

        with caplog.at_level(logging.WARNING, logger="strands_robots.device_connect._authz"):
            caplog.clear()
            result = asyncio.run(driver.step(n_steps=3, source_device="ctrl"))

        assert result["status"] == "success"
        assert sim.steps == 3
        assert bool([r for r in caplog.records if "SELF-ASSERTED" in r.getMessage()]) is insecure


class TestTheVocabularyIsSpelledOnce:
    """Which spellings opt in is decided in one place for the whole package."""

    def test_no_reader_of_the_variable_keeps_its_own_copy(self) -> None:
        """A second copy is how the readers could come to disagree about "yes".

        The population is derived: every module that names the variable is held
        to the rule, so a fourth reader is graded on arrival. ``STRANDS_MESH``
        already has this guard (``tests/mesh/test_mesh_env_vocabulary_has_one_owner``)
        and its own vocabulary happens to be the same three spellings - it is out
        of scope here because it never names this variable.
        """
        import ast
        import pathlib

        az = _authz()
        root = pathlib.Path(az.__file__).parents[2]
        readers = [
            path
            for path in sorted(root.joinpath("strands_robots").rglob("*.py"))
            if az._INSECURE_ENV in path.read_text()
        ]
        assert len(readers) >= 3, f"expected the resolver, the authorizer and the connector, got {readers}"
        spelled: list[str] = []
        for path in readers:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Tuple) and [e.value for e in node.elts if isinstance(e, ast.Constant)] == list(
                    az.INSECURE_TRUE
                ):
                    spelled.append(f"{path.relative_to(root)}:{node.lineno}")
        assert spelled == [f"strands_robots/device_connect/_authz.py:{_owner_line(az)}"], (
            f"the opt-in vocabulary is spelled {len(spelled)} times: {spelled}"
        )

    def test_the_resolver_parses_through_the_owner(self) -> None:
        """The resolver's env branch and the fallback cannot answer differently."""
        import strands_robots.device_connect as dc

        az = _authz()
        for spelling in (*az.INSECURE_TRUE, "TRUE", "Yes", "false", "on", "", None):
            assert dc.resolve_allow_insecure(None, spelling) is az.insecure_env_opts_in(spelling), spelling


def _owner_line(module: Any) -> int:
    """Line ``INSECURE_TRUE`` is assigned on, so the assertion names it."""
    import pathlib

    for i, line in enumerate(pathlib.Path(module.__file__).read_text().splitlines(), 1):
        if line.startswith("INSECURE_TRUE"):
            return i
    raise AssertionError("INSECURE_TRUE is not spelled in its owner")


class TestReadingTheRuntimeOffADriverDoesNotAssumeTheSetterRan:
    """``_device`` is created by ``set_device``, so an unattached driver has no attribute.

    ``DeviceDriver`` belongs to ``device_connect_edge`` and creates ``_device``
    in ``set_device`` rather than in ``__init__``, so an unattached driver
    presents as the attribute being *absent* - not as a ``None`` value. Passing
    ``self._device`` therefore raised ``AttributeError`` on every driver whose
    runtime never attached, which is precisely the case
    :func:`is_authorized_caller` documents the environment fallback for. The
    fallback was unreachable through a driver, so the two cells that graded it
    (``test_an_unattached_driver_still_falls_back_to_the_variable`` and the
    resolver's own row) both passed ``device=None`` to the helper directly and
    could not see it.

    That is a refusal the safety path cannot afford: the ``emergencyStop``
    handlers read the posture before deciding whether to honour a stop, and an
    ``AttributeError`` there is a stop that neither authorizes nor refuses.
    """

    def test_the_accessor_reports_absent_and_attached_alike(self) -> None:
        """Absent attribute reads as ``None``; after ``set_device`` it is the runtime."""
        az = _authz()

        class _Unattached:
            """A driver that has not been handed a runtime - no ``_device`` at all."""

        assert not hasattr(_Unattached(), "_device")
        assert az.attached_runtime(_Unattached()) is None

        runtime = _Runtime(True)

        class _Attached:
            _device = runtime

        assert az.attached_runtime(_Attached()) is runtime

    def test_an_rpc_on_a_never_attached_driver_decides_rather_than_raises(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The shape that regressed: a driver constructed and driven with no runtime.

        No ``set_device`` call, which is how every unit that exercises a driver
        directly builds one. The RPC has to reach the wrapped simulation, and
        the advisory has to follow the environment variable because that is the
        only source of a posture available.
        """
        from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

        az = _authz()

        class _Sim:
            def __init__(self) -> None:
                self.steps = 0

            def step(self, n_steps: int) -> dict[str, Any]:
                self.steps += n_steps
                return {"status": "success", "content": [{"text": f"stepped {n_steps}"}]}

        for env, expect_advisory in (("yes", True), ("no", False)):
            monkeypatch.setenv("DEVICE_CONNECT_ALLOW_INSECURE", env)
            az._warned_insecure_acl.clear()
            az._warned_permissive.clear()

            sim = _Sim()
            driver = SimulationDeviceDriver(sim)
            assert not hasattr(driver, "_device"), "set_device must not have run"

            with caplog.at_level(logging.WARNING, logger="strands_robots.device_connect._authz"):
                caplog.clear()
                result = asyncio.run(driver.step(n_steps=3, source_device="ctrl"))

            assert result["status"] == "success", env
            assert sim.steps == 3, env
            advisories = [r for r in caplog.records if "SELF-ASSERTED" in r.getMessage()]
            assert bool(advisories) is expect_advisory, env

    def test_no_authorization_call_reads_the_private_attribute_directly(self) -> None:
        """Derived over the drivers, so a call site added later is graded on arrival.

        Every ``is_authorized_caller`` call in the Device Connect drivers must
        reach the runtime through the accessor. ``self._device`` is refused here
        rather than left to the one driver-level cell that happened to construct
        an unattached driver, because the failure is per-call-site: a single new
        RPC spelling it the old way raises on exactly the bring-up shape this
        test file exists to cover.
        """
        import ast
        import pathlib

        import strands_robots.device_connect as dc

        package = pathlib.Path(dc.__file__).parent
        drivers = sorted(p for p in package.glob("*driver*.py"))
        assert len(drivers) >= 3, f"expected the three drivers, found {[p.name for p in drivers]}"

        checked: list[str] = []
        for path in drivers:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "id", None) != "is_authorized_caller":
                    continue
                passed = {kw.arg: kw.value for kw in node.keywords}
                where = f"{path.name}:{node.lineno}"
                assert "device" in passed, f"{where} decides the posture from the environment alone"
                assert ast.unparse(passed["device"]) == "attached_runtime(self)", (
                    f"{where} passes {ast.unparse(passed['device'])!r}; read the runtime through "
                    "attached_runtime(self), which tolerates a driver whose set_device never ran"
                )
                checked.append(where)

        assert len(checked) == 21, f"expected the 21 known call sites, graded {len(checked)}: {checked}"
