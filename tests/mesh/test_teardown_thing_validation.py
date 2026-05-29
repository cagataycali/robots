"""Regression test: teardown_thing validates thing_name before any AWS/FS call.

Addresses review thread on provision.py:320 (PR #228 R3) -- _validate_thing_name
was applied to provision_robot and provision_operator but NOT to teardown_thing,
leaving a path-traversal vector via ``DEFAULT_CERT_DIR / f"{thing_name}.pem"``.
"""

import pytest


class TestTeardownThingValidation:
    """teardown_thing must reject unsafe thing_name values."""

    def test_path_traversal_rejected(self):
        """thing_name containing '../' must raise ValueError before any I/O."""
        from strands_robots.mesh.iot.provision import teardown_thing

        with pytest.raises(ValueError, match="invalid characters"):
            teardown_thing("../../etc/passwd")

    def test_dots_rejected(self):
        """thing_name containing '.' must raise ValueError."""
        from strands_robots.mesh.iot.provision import teardown_thing

        with pytest.raises(ValueError, match="invalid characters"):
            teardown_thing("robot.v2")

    def test_colons_rejected(self):
        """thing_name containing ':' must raise ValueError."""
        from strands_robots.mesh.iot.provision import teardown_thing

        with pytest.raises(ValueError, match="invalid characters"):
            teardown_thing("robot:alpha")

    def test_empty_rejected(self):
        """Empty thing_name must raise ValueError."""
        from strands_robots.mesh.iot.provision import teardown_thing

        with pytest.raises(ValueError, match="non-empty string"):
            teardown_thing("")

    def test_valid_name_passes_validation(self, monkeypatch):
        """Valid thing_name passes validation, reaches boto3 import."""
        from strands_robots.mesh.iot import provision

        # Mock _require_boto3 to avoid real AWS calls
        mock_called = []

        def fake_require_boto3():
            mock_called.append(True)
            raise ImportError("boto3 not available in test")

        monkeypatch.setattr(provision, "_require_boto3", fake_require_boto3)

        with pytest.raises(ImportError, match="boto3 not available"):
            provision.teardown_thing("valid-robot-name_123")

        assert mock_called, "_require_boto3 should be called after validation passes"
