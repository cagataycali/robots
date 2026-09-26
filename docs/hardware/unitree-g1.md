---
description: Unitree G1 bring-up over CycloneDDS - what connect_eagerly subscribes, the FSM-gated motion bundle, and installing unitree_sdk2py on x86_64 and aarch64.
---

# Unitree G1 over CycloneDDS

The G1 has no lerobot robot type either, so `mode="real"` builds the native
CycloneDDS driver in `strands_robots.drivers.g1` (the registry declares
`hardware.driver = "strands"`):

```python
from strands_robots import Robot

g1 = Robot("g1", mode="real", port="192.168.123.161")   # network_interface="eth0" by default
g1.connect_eagerly()      # subscribes rt/lowstate, bms, lidar, mainboard; None when the bus is up
await g1.get_status()     # connection, FSM, battery
```

The driver-as-tool is deliberately small - `sensors`, `status`, `stop` - so an
agent can introspect the robot the day it is built. Motion goes through the
FSM-gated `g1_tools` bundle (`g1_send_action`, `g1_run_policy`, `g1_task` over
the control-loop lifecycle, the `g1_safe_*` posture verbs) and, for the raw SDK, `use_unitree`; see the
[hardware tools](tools.md) and [security](../security.md) pages.

## Installing the Unitree SDK

`unitree_sdk2py` is Unitree's vendor SDK and is **not** an extra of this
project. It cannot honestly be one: the PyPI `unitree-sdk2` 1.0.1 wheel ships no
`g1` or `comm` package (its `__init__` imports a `b2` it does not contain, so
`import unitree_sdk2py` fails) and pins `cyclonedds==0.10.2`, whose wheels stop
at Python 3.10 - under this project's `requires-python = ">=3.12"` that is a
source build that wants the CycloneDDS C library. Without the SDK the driver
still imports and builds; `connect_eagerly()` and every write verb return a
refusal that names this recipe.

A *partial* install fails elsewhere, and that is the shape the PyPI wheel
produces. With the bus bindings and the IDL types present but no `comm`
package, `connect_eagerly()` **succeeds** and the only thing that fails is the
motion-switcher open - reported as `motion_switcher_open_error` by the G1's
`get_status()`, and as the refusal from the Go2's `release_sport_mode()`. Both
name the same recipe and keep the SDK's own exception, so the module that is
actually missing is in the text.

The upstream checkout installed beside a `cyclonedds` wheel is what works, and
the binding comes from this project's `[ros2]` extra - the one place the range is
declared. On macOS arm64 and x86_64 Linux (Python 3.12):

```bash
pip install 'strands-robots[ros2]'                # the cyclonedds binding
git clone https://github.com/unitreerobotics/unitree_sdk2_python
pip install --no-deps -e ./unitree_sdk2_python    # --no-deps skips the ==0.10.2 pin
python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('ok')"
```

On Linux aarch64 - the Jetson the robot ships with - `cyclonedds` publishes no
wheel at any version, so the C library comes first and the binding is built
against it:

```bash
git clone --branch 0.10.2 --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cdds
cmake -S /tmp/cdds -B /tmp/cdds/build -DCMAKE_BUILD_TYPE=Release && sudo cmake --build /tmp/cdds/build --target install
export CYCLONEDDS_HOME=/usr/local
pip install 'cyclonedds==0.10.2'
git clone https://github.com/unitreerobotics/unitree_sdk2_python
pip install --no-deps -e ./unitree_sdk2_python
```

0.10.2 is the CycloneDDS release Unitree's own images and SDK pin, and the one
the `g1_tools` bundle was developed against. The 11.x wheel imports, builds the
IDL types and binds a `ChannelFactory`, but it has not been proven against a
live G1 bus; if the robot's topics stay silent under 11.x, build 0.10.2 as
above. Point `CYCLONEDDS_URI` at the robot's `cyclonedds.xml` when the default
multicast discovery does not find it.

To list a service's methods `use_unitree` reads the SDK's *source* rather than
importing it, searching `UNITREE_SDK_PATH`, then `/tmp/unitree_sdk2_python`,
then a checkout beside the installed package. It is a search root, not an
import path - the install above is what makes the SDK importable.

## See also

- [Native drivers](native-drivers.md) - the contract this driver satisfies.
- [Humanoids](../robots/humanoids.md) - the catalog entry, and the family's
  other native-driver bring-ups.
- [Hardware tools](tools.md) - the `g1_tools` bundle and `use_unitree`.
- [Mobile](../robots/mobile.md) - the Go2 shares this SDK and this recipe.
- [Robot factory](../getting-started/robot-factory.md) - every `Robot()` kwarg.
