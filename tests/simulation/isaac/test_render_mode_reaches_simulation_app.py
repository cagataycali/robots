"""render_mode reaches SimulationApp, and the abandoned renderer helpers stay deleted.

Pins the two halves of #2324:

1. ``IsaacConfig.render_mode`` used to be stored and gate a few code paths
   (headless errors, warmup, native-res upscale) but was never forwarded to
   the ``SimulationApp`` launch config, so ``rtx_pathtracing`` silently ran
   the default real-time renderer -- the silently-dropped-kwarg pattern the
   review learnings forbid. ``create_world`` now maps the mode to the
   documented ``renderer`` launch key (``rtx_realtime`` ->
   ``RayTracedLighting``, ``rtx_pathtracing`` -> ``PathTracing``,
   ``headless`` -> no key), and ``_get_or_create_simulation_app`` reports a
   launch request the existing process-wide singleton cannot honour instead
   of silently returning an app configured otherwise.

2. ``_configure_renderer`` (carb-settings DLSS mitigation that RTX re-asserted
   away every render tick) and the backend ``_add_lighting`` (shadowed by the
   example's own) were dead code and are deleted per repo rule 10. If either
   name returns, it must return *wired* -- called from a production path --
   and this pin replaced with a test of that call.

All tests here run without Isaac Sim installed: they exercise the launch
plumbing up to (and including) the launch-config dict, never a real Kit boot.
"""

from __future__ import annotations

import gc
import logging

import pytest

from strands_robots.simulation.isaac import simulation as isaac_simulation
from strands_robots.simulation.isaac.config import RENDER_MODES
from strands_robots.simulation.isaac.simulation import IsaacSimulation


@pytest.fixture()
def fresh_singleton(monkeypatch):
    """Reset the process-wide SimulationApp singleton bookkeeping per test."""
    monkeypatch.setattr(isaac_simulation, "_SIMULATION_APP", None)
    monkeypatch.setattr(isaac_simulation, "_SIMULATION_APP_LAUNCH", None)


class _RecordingApp:
    """Stands in for ``SimulationApp``: records the launch dict it was given."""

    def __init__(self, launch: dict):
        self.launch = dict(launch)


@pytest.fixture()
def fake_isaacsim(monkeypatch, fresh_singleton):
    """Inject a fake ``isaacsim`` module whose SimulationApp records its config."""
    import sys
    import types

    mod = types.ModuleType("isaacsim")
    mod.SimulationApp = _RecordingApp
    monkeypatch.setitem(sys.modules, "isaacsim", mod)
    return mod


class TestCreateWorldForwardsRenderMode:
    """create_world derives the SimulationApp launch config from render_mode."""

    def test_every_render_mode_makes_a_renderer_decision(self):
        """Stated over the whole enumeration rather than three representative values.

        ``__post_init__`` admits exactly ``RENDER_MODES``, and a member with no
        entry in ``_RENDERER_BY_MODE`` reaches ``.get()`` as ``None`` and
        silently launches Kit's default renderer -- the #2324 defect this file
        pins, one mode over. Only ``"headless"`` may select nothing, and it
        does so deliberately (documented at the map and at the config field).
        A new mode must therefore either gain a renderer entry or be added to
        the explicit no-renderer set here, on purpose.
        """
        mapped = set(isaac_simulation._RENDERER_BY_MODE)
        assert set(RENDER_MODES) == mapped | {"headless"}, (
            "a render_mode with no entry in _RENDERER_BY_MODE silently launches Kit's "
            "default renderer -- the #2324 defect this file pins"
        )

    @pytest.mark.parametrize(
        ("render_mode", "expected_launch_config"),
        [
            ("headless", None),
            ("rtx_realtime", {"renderer": "RayTracedLighting"}),
            ("rtx_pathtracing", {"renderer": "PathTracing"}),
        ],
    )
    def test_render_mode_selects_renderer_launch_key(self, monkeypatch, render_mode, expected_launch_config):
        recorded: dict = {}

        def _record(headless=True, launch_config=None, **kwargs):
            recorded["headless"] = headless
            recorded["launch_config"] = launch_config
            # Refuse past the point under test: create_world turns this into
            # its structured error envelope, which the assertion below pins.
            raise ImportError("stop after recording the launch request")

        monkeypatch.setattr(isaac_simulation, "_get_or_create_simulation_app", _record)
        sim = IsaacSimulation(num_envs=1, headless=True, render_mode=render_mode)
        result = sim.create_world()

        assert result["status"] == "error"  # ImportError -> structured envelope
        assert recorded["headless"] is True
        assert recorded["launch_config"] == expected_launch_config


class _LeftToTheCollector:
    """An object whose finalizer logs on this module's logger.

    The shape :meth:`~strands_robots.simulation.base.SimEngine.__del__` takes for
    an ``IsaacSimulation`` a test built and never destroyed: ``cleanup()`` ->
    ``destroy()``, which on a fake ``isaacsim`` logs ``World cleanup warning``
    and ``Stage clear warning`` at WARNING. The self-reference keeps it out of
    refcount teardown so it goes when the cyclic collector runs, which is when
    the real ones go.
    """

    def __init__(self) -> None:
        self.cycle = self

    def __del__(self) -> None:
        isaac_simulation.logger.warning("World cleanup warning: %s", "left to the collector")


def _launch_key_reports(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The WARNING records the launch path emits, and nothing else on the logger.

    ``caplog.records`` is every record the call phase captured, on every
    logger, and ``isaac_simulation.logger`` is shared with ``destroy()``: an
    engine another test left to the cyclic collector is finalised at whatever
    point the collector fires, and on CI run 36098888798 (gw0, ``--dist
    loadfile``) four of them fired inside
    ``test_headless_only_second_request_is_silent``, whose ``assert not
    [WARNING records]`` then failed on a launch that had dropped nothing. The
    report this class grades names the launch keys it could not apply, so
    read that. The wording is pinned by the differing-request test, which
    asserts exactly one such report, so a rewording cannot make the silent
    tests vacuous without failing that one.
    """
    return [r for r in caplog.records if r.levelno == logging.WARNING and "launch keys" in r.getMessage()]


class TestSimulationAppLaunch:
    """_get_or_create_simulation_app plumbing around the create-once singleton."""

    def test_renderer_reaches_the_simulation_app_launch_dict(self, fake_isaacsim):
        app = isaac_simulation._get_or_create_simulation_app(headless=True, launch_config={"renderer": "PathTracing"})
        assert app.launch == {"renderer": "PathTracing", "headless": True}

    def test_differing_second_request_warns_and_returns_the_existing_app(self, fake_isaacsim, caplog):
        first = isaac_simulation._get_or_create_simulation_app(
            headless=True, launch_config={"renderer": "RayTracedLighting"}
        )
        with caplog.at_level(logging.WARNING, logger=isaac_simulation.logger.name):
            second = isaac_simulation._get_or_create_simulation_app(
                headless=True, launch_config={"renderer": "PathTracing"}
            )
        assert second is first
        reports = _launch_key_reports(caplog)
        assert len(reports) == 1
        assert "renderer" in reports[0].getMessage()
        assert "PathTracing" in reports[0].getMessage()

    def test_matching_second_request_is_silent(self, fake_isaacsim, caplog):
        first = isaac_simulation._get_or_create_simulation_app(headless=True, launch_config={"renderer": "PathTracing"})
        with caplog.at_level(logging.WARNING, logger=isaac_simulation.logger.name):
            second = isaac_simulation._get_or_create_simulation_app(
                headless=True, launch_config={"renderer": "PathTracing"}
            )
        assert second is first
        assert not _launch_key_reports(caplog)

    def test_headless_only_second_request_is_silent(self, fake_isaacsim, caplog):
        # A render_mode="headless" sim coming up after an RTX one selects no
        # renderer, so it drops nothing and must not warn.
        isaac_simulation._get_or_create_simulation_app(headless=True, launch_config={"renderer": "PathTracing"})
        with caplog.at_level(logging.WARNING, logger=isaac_simulation.logger.name):
            isaac_simulation._get_or_create_simulation_app(headless=True)
        assert not _launch_key_reports(caplog)

    def test_a_finalizer_on_the_shared_logger_is_not_a_launch_report(self, fake_isaacsim, caplog):
        # The failure shape of CI run 36098888798: an engine another test left
        # to the cyclic collector is finalised inside this test's window and
        # logs on the same logger. That record is not a dropped launch key, so
        # the silent-request verdict must not read it as one; the control
        # asserts the record did arrive, so a green here is the filter and not
        # the collector's timing.
        isaac_simulation._get_or_create_simulation_app(headless=True, launch_config={"renderer": "PathTracing"})
        _LeftToTheCollector()
        with caplog.at_level(logging.WARNING, logger=isaac_simulation.logger.name):
            gc.collect()
            isaac_simulation._get_or_create_simulation_app(headless=True)
        assert [r for r in caplog.records if r.getMessage().startswith("World cleanup warning")]
        assert not _launch_key_reports(caplog)


class TestAbandonedRendererHelpersStayDeleted:
    """Repo rule 10: no dead code. See the module docstring for the wire-or-delete terms."""

    @pytest.mark.parametrize("name", ["_configure_renderer", "_add_lighting"])
    def test_deleted_helper_is_gone(self, name):
        assert not hasattr(IsaacSimulation, name), (
            f"IsaacSimulation.{name} was deleted as dead code (#2324); re-adding it "
            f"requires a production call site and a test of that call in place of this pin."
        )
