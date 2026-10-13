import importlib
import sys


def test_the_package_imports_with_nothing_heavy() -> None:
    heavy = ("mujoco", "torch", "lerobot", "zenoh", "isaacsim", "warp")
    for name in [m for m in sys.modules if m.split(".")[0] in heavy]:
        del sys.modules[name]
    pkg = importlib.import_module("strands_robots")
    assert pkg.__version__.startswith("1.0.0")
    assert not [m for m in sys.modules if m.split(".")[0] in heavy]
