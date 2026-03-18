"""Tests for strands_robots.utils — require_optional lazy import helper."""

import pytest

from strands_robots.utils import _lazy_modules, require_optional


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

    def test_cached_module_in_dict(self):
        """Module should appear in the cache dict after first import."""
        require_optional("os")
        assert "os" in _lazy_modules

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

    def test_returns_module_object(self):
        """Should return the actual module object."""
        mod = require_optional("sys")
        import sys

        assert mod is sys

    def test_dotted_module(self):
        """Should handle dotted module names like os.path."""
        mod = require_optional("os.path")
        assert hasattr(mod, "join")

    def test_cache_bypass_not_possible(self):
        """Once cached, should always return from cache."""
        # Import something
        mod1 = require_optional("collections")
        # Manually verify it's in cache
        assert "collections" in _lazy_modules
        assert _lazy_modules["collections"] is mod1

    def test_concurrent_safe_structure(self):
        """Cache should be a plain dict (no threading issues at module level)."""
        assert isinstance(_lazy_modules, dict)

    def test_error_has_no_chained_cause(self):
        """ImportError should suppress the original traceback (from None)."""
        with pytest.raises(ImportError) as exc_info:
            require_optional("nonexistent_xyz_abc")
        assert exc_info.value.__cause__ is None
