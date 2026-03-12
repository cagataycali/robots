"""Tests for strands_robots top-level package imports."""


class TestPackageImport:
    def test_import_package(self):
        import strands_robots

        assert hasattr(strands_robots, "__all__")

    def test_core_exports(self):
        from strands_robots import (
            MockPolicy,
            Policy,
            Robot,
            create_policy,
            list_providers,
            list_robots,
            resolve_policy,
        )

        assert Robot is not None
        assert Policy is not None
        assert MockPolicy is not None
        assert callable(create_policy)
        assert callable(list_providers)
        assert callable(list_robots)
        assert callable(resolve_policy)

    def test_all_is_list(self):
        import strands_robots

        assert isinstance(strands_robots.__all__, list)
        assert len(strands_robots.__all__) > 0

    def test_submodule_imports(self):
        """Advanced users import from submodules, not top-level."""
        from strands_robots.motion_library import Motion, MotionLibrary

        assert MotionLibrary is not None
        assert Motion is not None

        from strands_robots.telemetry import EventCategory, TelemetryStream

        assert TelemetryStream is not None
        assert EventCategory is not None

    def test_clean_namespace(self):
        """Top-level should be small — configs/internals live in submodules."""
        import strands_robots

        # No config classes at top level
        for name in strands_robots.__all__:
            if name.endswith("Config") and name not in ("TrainConfig",):
                raise AssertionError(
                    f"Config class at top level: {name} — "
                    f"should be imported from its submodule"
                )

    def test_create_trainer(self):
        """create_trainer should be a top-level export."""
        from strands_robots import create_trainer

        assert callable(create_trainer)
