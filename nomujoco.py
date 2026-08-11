"""pytest plugin: emulate an install without the mujoco package."""
import sys


def pytest_configure(config):
    for key in [m for m in sys.modules if m == "mujoco" or m.startswith("mujoco.")]:
        del sys.modules[key]
    sys.modules["mujoco"] = None  # `import mujoco` -> ModuleNotFoundError
