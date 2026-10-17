import importlib
import subprocess
import sys


def test_the_package_imports_with_nothing_heavy() -> None:
    heavy = ("mujoco", "torch", "lerobot", "zenoh", "isaacsim", "warp", "onnxruntime", "huggingface_hub")
    for name in [m for m in sys.modules if m.split(".")[0] in heavy]:
        del sys.modules[name]
    pkg = importlib.import_module("strands_robots")
    assert pkg.__version__.startswith("1.0.0")
    assert not [m for m in sys.modules if m.split(".")[0] in heavy]


def test_simulation_without_mujoco_says_how_to_install() -> None:
    probe = (
        "import sys; sys.modules['mujoco'] = None\n"
        "from strands_robots import Robot\n"
        "try:\n    Robot('so101')\nexcept ImportError as e:\n    print(e)\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.startswith("Simulation() needs MuJoCo: pip install 'strands-robots[mujoco]'"), out
