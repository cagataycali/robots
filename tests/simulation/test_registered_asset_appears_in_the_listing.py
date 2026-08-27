"""A ``register_urdf`` asset is named by the listing that advertises it.

``MuJoCoSimEngine.list_urdfs`` is the discovery entry point an agent uses to
learn what it can spawn, and its docstring promises "built-in *and*
user-registered". It returns
:func:`strands_robots.simulation.model_registry.list_available_models`, which
promises "Menagerie + custom" and used to return the asset-manager table
alone whenever the asset manager was importable - which is every normal
install. So the two halves were an either/or, and the half that was dropped
was the one the caller had just written:
``register_urdf`` reported ``Registered ... Resolved: <path>``,
``resolve_urdf`` resolved it, ``add_robot`` spawned it, and the listing denied
it existed. ``add_robot``'s own unresolved-model message sends the caller to
``list_urdfs`` to "pick a registered model", so the recovery path pointed at
the listing that omitted the registration.

The custom half was graded only through the asset-manager-*absent* branch
(``monkeypatch.setattr(mr, "_HAS_ASSET_MANAGER", False)``), a branch that never
executes in a real install, and the present-branch test asserted only that the
built-in columns appear. That is why the suite was green.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import strands_robots.simulation.model_registry as mr

_SECTION = "Registered URDFs:"


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every case its own runtime registry.

    ``_URDF_REGISTRY`` is a module global that ``register_urdf`` mutates and
    that no shipped test clears, so without this a case would read whatever
    earlier files registered.
    """
    monkeypatch.setattr(mr, "_URDF_REGISTRY", {})


def _asset(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.xml"
    path.write_text("<mujoco/>")
    return path


class TestThePremise:
    """The configuration under test is the one that ships."""

    def test_the_asset_manager_is_importable_here(self) -> None:
        """Otherwise these cases would grade the fallback branch instead."""
        assert mr._HAS_ASSET_MANAGER is True

    def test_a_registered_asset_really_resolves(self, tmp_path: Path) -> None:
        """The listing is the only surface that disagreed - resolution worked."""
        asset = _asset(tmp_path, "premise_arm")
        mr.register_urdf("premise_arm", str(asset))

        assert mr.resolve_urdf("premise_arm") == str(asset)
        assert mr.list_registered_urdfs()["premise_arm"] == str(asset)


class TestARegisteredAssetIsNamedByTheListing:
    """The regression: what was written is what is read back."""

    def test_the_listing_names_a_registered_asset(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path, "widget_arm")
        mr.register_urdf("widget_arm", str(asset))

        out = mr.list_available_models()

        assert "widget_arm" in out
        assert f"[OK] widget_arm: {asset}" in out

    def test_a_registered_asset_that_does_not_resolve_is_marked_missing(self) -> None:
        """A dangling registration is reported, not dropped - it is the likelier typo."""
        mr.register_urdf("ghost_arm", "/no/such/file.xml")

        out = mr.list_available_models()

        assert "[MISSING] ghost_arm: /no/such/file.xml" in out

    def test_every_registered_asset_is_named_not_just_the_first(self, tmp_path: Path) -> None:
        for name in ("alpha_arm", "beta_arm", "gamma_arm"):
            mr.register_urdf(name, str(_asset(tmp_path, name)))

        out = mr.list_available_models()

        assert [n for n in ("alpha_arm", "beta_arm", "gamma_arm") if n not in out] == []

    def test_the_registered_section_follows_the_built_in_table(self, tmp_path: Path) -> None:
        """Appended, not prepended - the built-in table stays the listing's lead."""
        mr.register_urdf("widget_arm", str(_asset(tmp_path, "widget_arm")))

        out = mr.list_available_models()

        assert out.index("Name") < out.index(_SECTION)

    def test_the_engine_discovery_surface_names_it(self, tmp_path: Path) -> None:
        """``list_urdfs`` is the agent-facing route, and the one whose docstring promises the union."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation import create_simulation

        asset = _asset(tmp_path, "engine_arm")
        # ``create_simulation`` is annotated ``SimEngine``; the discovery pair
        # under test is declared on the MuJoCo engine, not on the base class.
        sim: Any = create_simulation("mujoco")
        assert sim.register_urdf("engine_arm", str(asset))["status"] == "success"

        text = sim.list_urdfs()["content"][0]["text"]

        assert "engine_arm" in text


class TestTheBuiltInListingIsUnchanged:
    """Naming the registered assets must not cost the built-in table."""

    def test_the_built_in_table_is_still_reported(self, tmp_path: Path) -> None:
        mr.register_urdf("widget_arm", str(_asset(tmp_path, "widget_arm")))

        out = mr.list_available_models()

        assert "Name" in out and "Category" in out

    def test_an_empty_registry_adds_no_section(self) -> None:
        """A default install's listing is byte-for-byte what it was."""
        assert _SECTION not in mr.list_available_models()

    def test_a_name_that_was_never_registered_is_not_claimed(self, tmp_path: Path) -> None:
        mr.register_urdf("widget_arm", str(_asset(tmp_path, "widget_arm")))

        assert "unregistered_arm" not in mr.list_available_models()

    def test_the_fallback_branch_still_reports_both_states(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the asset manager the registered section is the whole listing."""
        mr.register_urdf("present_arm", str(_asset(tmp_path, "present_arm")))
        mr.register_urdf("absent_arm", "/no/such/file.xml")
        monkeypatch.setattr(mr, "_HAS_ASSET_MANAGER", False)

        out = mr.list_available_models()

        assert out.startswith(_SECTION)
        assert "[OK] present_arm" in out
        assert "[MISSING] absent_arm" in out


class TestOneVocabularyForTheRegisteredSection:
    """Both branches render the rows through one helper, so they cannot drift."""

    def test_the_row_format_is_written_down_once(self) -> None:
        source = Path(mr.__file__).read_text(encoding="utf-8")

        assert source.count('"[OK]"') == 1
        assert source.count('"[MISSING]"') == 1

    def test_the_listing_renders_no_row_of_its_own(self) -> None:
        import inspect

        body = inspect.getsource(mr.list_available_models)

        assert "[OK]" not in body
        assert "[MISSING]" not in body
        assert "_registered_urdf_lines()" in body
