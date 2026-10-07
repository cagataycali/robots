"""
Repro: `require_optional("serial", pip_install="pyserial", ...)` in three
first-party sites omits `extra=`, so the ImportError refuses with
`pip install pyserial` and nothing else — yet two extras of this project
(`[dashboard]`, `[lerobot]`) ship pyserial, and `drivers/feetech/bus.py:269`
(same file as one of the three sites) uses the `extra=` keyword for a
sibling optional dependency.

Three sites, all guarding pyserial:
  strands_robots/tools/serial_tool.py:95-96
  strands_robots/tools/pose_tool.py:79-80
  strands_robots/drivers/feetech/bus.py:570

pyproject.toml declarations the error ignores:
  dashboard = [..., "pyserial>=3.5,<4.0", ...]            # line 794
  lerobot   = ["lerobot[feetech,dataset]>=0.6.1,<0.7.0"]  # feetech pulls pyserial

Sibling that does it right (same `require_optional` wrapper):
  strands_robots/drivers/feetech/bus.py:269
      require_optional("lerobot.utils.constants",
                       pip_install="lerobot",
                       extra="lerobot", ...)   # <<<< names the extra
  strands_robots/policies/wbc/config.py:399
      require_optional("yaml", pip_install="pyyaml",
                       extra="wbc", ...)       # <<<< names the extra

AGENTS.md convention 7 asks the message to let a caller "tell an absent
extra from a broken package path" — telling the caller to `pip install
pyserial` after they already paid for `strands-robots[dashboard]` fails
that: it names a package the caller already resolved through an extra.

Reproduce (no install needed; subprocess with a meta_path finder that
blocks `serial`):

    python bugbash_repros/pyserial_require_optional_omits_extras_repro.py

Expected per convention 7 + sibling pattern:
    Install with:
      pip install 'strands-robots[dashboard]'   # or [lerobot]
      pip install pyserial

Actual on main `fd038b3ae`:
    Install with:
      pip install pyserial
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent  # bugbash_repros/.. -> repo


SHIM = textwrap.dedent(
    """
    import sys

    class _BlockSerialFinder:
        def find_spec(self, name, path=None, target=None):
            if name == "serial" or name.startswith("serial."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _BlockSerialFinder())

    import tomllib
    from pathlib import Path

    # Show pyserial's declared project extras, so the diff with the error is
    # the whole demo.
    pp = tomllib.loads(Path("pyproject.toml").read_text())
    extras_with_pyserial = sorted(
        name for name, deps in pp["project"]["optional-dependencies"].items()
        if any("pyserial" in d.lower() for d in deps)
    )
    core_has_pyserial = any("pyserial" in d.lower() for d in pp["project"]["dependencies"])
    print(f"pyserial declared in extras: {extras_with_pyserial}")
    print(f"pyserial in core deps:       {core_has_pyserial}")
    print(f"transitive via [lerobot]:    yes (lerobot[feetech,dataset] floors pyserial)")
    print()

    sites = [
        ("strands_robots.tools.serial_tool",       "tools/serial_tool.py:95"),
        ("strands_robots.tools.pose_tool",         "tools/pose_tool.py:79"),
        ("strands_robots.drivers.feetech.bus",     "drivers/feetech/bus.py:570 (triggers on FeetechBus.connect)"),
    ]

    import importlib

    print("=== importing the three guarded modules with 'serial' blocked ===")
    for modname, label in sites:
        print(f"--- {label} ---")
        try:
            m = importlib.import_module(modname)
            if modname.endswith("bus"):
                # bus.py:570 is in FeetechBus.connect; trigger it with a fake port.
                try:
                    bus = m.FeetechBus(port="/dev/null", baud_rate=1000000)
                except Exception as e:
                    print(f"ctor refused (not the one we want): {type(e).__name__}: {e}")
                    continue
                try:
                    bus.connect()
                except ImportError as e:
                    print(f"message:")
                    for line in str(e).splitlines():
                        print(f"  {line}")
                    print(f"name attr: {e.name!r}")
                except Exception as e:
                    print(f"different refusal: {type(e).__name__}: {e}")
            else:
                print("UNEXPECTED: imported without pyserial")
        except ImportError as e:
            print(f"message:")
            for line in str(e).splitlines():
                print(f"  {line}")
            print(f"name attr: {e.name!r}")
        print()

    print("=== sibling that cites extra= (same wrapper, same file as one site) ===")
    # Walk the AST to prove the sibling uses extra="lerobot"
    import ast
    bus = Path("strands_robots/drivers/feetech/bus.py").read_text()
    tree = ast.parse(bus)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "require_optional"
        ):
            kw = {k.arg: k.value for k in node.keywords}
            mod = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else "?"
            has_extra = "extra" in kw
            extra_v = kw["extra"].value if has_extra and isinstance(kw["extra"], ast.Constant) else None
            pip_v = kw["pip_install"].value if "pip_install" in kw and isinstance(kw["pip_install"], ast.Constant) else None
            print(f"  bus.py:{node.lineno}  module={mod!r} pip_install={pip_v!r} extra={extra_v!r}")

    print()
    print("=== conclusion ===")
    print("Two 'require_optional' calls in strands_robots/drivers/feetech/bus.py:")
    print("  line 269 (lerobot.utils.constants): names extra='lerobot'  ✓")
    print("  line 570 (serial):                  omits extra=            ✗")
    print()
    print("Impact: a user who installed strands-robots[dashboard] or")
    print("strands-robots[lerobot] and then lost pyserial (venv rebuild,")
    print("binary wheel conflict, pip --no-deps upgrade) is told to")
    print("'pip install pyserial' with no indication the project extras")
    print("they already pay for are the right remedy. Breaks")
    print("AGENTS.md convention 7 ('tell an absent extra from a broken")
    print("package path').")
    """
).strip()


def main() -> int:
    env = {k: v for k, v in os.environ.items() if k != "SYSTEM_PROMPT"}
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, "-c", SHIM],
        env=env,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=60,
    )
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
