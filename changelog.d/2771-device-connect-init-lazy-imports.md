### Fixed: `strands_robots.device_connect` imports on a stock `pip install strands-robots`

The package's `__init__` opened by doing `from device_connect_edge import DeviceRuntime`
unconditionally, and `device-connect-edge` lives only in the `[device-connect]` /
`[all]` extras. Sitting next to it in the same package,
`strands_robots.device_connect.reachy_transport` is stdlib-only; the native
Reachy driver imports that leaf on every daemon touch. Importing the leaf
executes the parent package's `__init__`, so on a default install every
`ReachyDriver` call reached `_daemon_get` / `_wire_commands` and raised
`ModuleNotFoundError: No module named 'device_connect_edge'` -- escaping the
driver's own no-raise refusal contract and breaking three `AGENTS.md`
conventions at once (`Return error dicts, never raise`, `require_optional()`
for optional deps, the module-load discipline the driver's docstring makes
explicit). CI never saw it because the hatch env installs `[all]`.

`__init__` now uses PEP 562 `__getattr__` (documented at
[python.org](https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access))
to resolve every public name from the export table on first access. Names that
need the extra (`init_device_connect`, `init_device_connect_sync`,
`resolve_allow_insecure`, `RobotDeviceDriver`, `SimulationDeviceDriver`,
`ReachyMiniDriver`) live in a private `_impl` submodule that is imported only
when a caller reaches for one of them, so a stock install can import
`strands_robots.device_connect.reachy_transport` and reach the Reachy driver
without paying the extra. The public surface is unchanged and each name still
raises a `ModuleNotFoundError` naming `device_connect_edge` if used without the
extra, so a caller that needs those symbols is still told which extra to
install -- the failure just no longer arrives as collateral damage on a caller
that never asked for them.

Two behaviours are preserved on purpose: `dir(strands_robots.device_connect)`
still advertises every public name (via a `__dir__` override) so IDE completion
sees the same surface as before, and each resolved name is cached on the
package so identity assertions like
`isinstance(x, RobotDeviceDriver)` and `RobotDeviceDriver is
strands_robots.device_connect.RobotDeviceDriver` hold across calls. Fixes
#2771.
