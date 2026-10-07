"""The registry package re-exports every public function of its discovery module."""

import inspect

import pytest

import strands_robots.registry as registry
from strands_robots.registry import discovery

# ``invalidate_cache`` is owned by the loader's broader cache reset.
_NOT_REEXPORTED = {"invalidate_cache"}

_DISCOVERY_FUNCTIONS = sorted(
    name
    for name, fn in inspect.getmembers(discovery, inspect.isfunction)
    if fn.__module__ == discovery.__name__ and not name.startswith("_") and name not in _NOT_REEXPORTED
)


@pytest.mark.parametrize("name", _DISCOVERY_FUNCTIONS)
def test_discovery_function_is_importable_from_the_registry(name):
    assert name in registry.__all__
    assert getattr(registry, name) is getattr(discovery, name)
