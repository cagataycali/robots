"""Tests for strands_robots.utils — require_optional lazy import helper."""

import pytest

from strands_robots.utils import require_optional


class TestRequireOptional:
    """Tests for the require_optional lazy import utility."""

    def test_imports_stdlib_module(self):
        """Should successfully import a stdlib module."""
        mod = require_optional("json")
        assert hasattr(mod, "dumps")

    def test_caches_module(self):
        """Second call should return the cached module (same object)."""
        mod1 = require_optional("json")
        mod2 = require_optional("json")
        assert mod1 is mod2

    def test_missing_module_raises_import_error(self):
        """Non-existent module should raise ImportError."""
        with pytest.raises(ImportError):
            require_optional("nonexistent_module_xyz_12345")

    def test_error_message_includes_module_name(self):
        with pytest.raises(ImportError, match="nonexistent_module_xyz"):
            require_optional("nonexistent_module_xyz")

    def test_error_message_includes_purpose(self):
        with pytest.raises(ImportError, match="for testing"):
            require_optional("nonexistent_xyz", purpose="testing")

    def test_error_message_includes_pip_install(self):
        with pytest.raises(ImportError, match="pip install my-package"):
            require_optional("nonexistent_xyz", pip_install="my-package")

    def test_error_message_includes_extra(self):
        with pytest.raises(ImportError, match="strands-robots\\[my-extra\\]"):
            require_optional("nonexistent_xyz", extra="my-extra")

    def test_error_message_default_pip_install(self):
        """When pip_install is not set, should use module_name."""
        with pytest.raises(ImportError, match="pip install nonexistent_xyz"):
            require_optional("nonexistent_xyz")

    def test_dotted_module(self):
        """Should handle dotted module names like os.path."""
        mod = require_optional("os.path")
        assert hasattr(mod, "join")


# ── safe_join / get_search_paths tests (added for PR #84 follow-up) ──


class TestSafeJoin:
    """Tests for the centralised path-traversal guard."""

    def test_joins_clean_paths(self, tmp_path):
        from strands_robots.utils import safe_join

        result = safe_join(tmp_path, "robot/model.xml")
        assert result == tmp_path / "robot" / "model.xml"

    def test_rejects_traversal(self, tmp_path):
        from strands_robots.utils import safe_join

        with pytest.raises(ValueError, match="Path traversal blocked"):
            safe_join(tmp_path, "../etc/passwd")

    def test_rejects_absolute_escape(self, tmp_path):
        from strands_robots.utils import safe_join

        with pytest.raises(ValueError, match="Path traversal blocked"):
            safe_join(tmp_path, "robot/../../etc/passwd")

    def test_same_path_is_allowed(self, tmp_path):
        from strands_robots.utils import safe_join

        # Empty / dot path resolves to base itself — must not raise
        result = safe_join(tmp_path, ".")
        assert result == tmp_path


class TestGetSearchPaths:
    """Tests for the centralised search-path resolver."""

    def test_returns_assets_dir_and_cwd_assets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        from strands_robots.utils import get_search_paths

        paths = get_search_paths()
        assert tmp_path in paths
        assert (tmp_path / "assets") in paths

    def test_returns_unique_paths(self, tmp_path, monkeypatch):
        # When CWD is already the assets dir, we shouldn't list the same path twice
        # (deduping is explicit in the implementation).
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
        from strands_robots.utils import get_search_paths

        paths = get_search_paths()
        assert len(paths) == len(set(paths))
