### Fixed: the Isaac launch tests read the launch report, not every warning on the logger

`TestSimulationAppLaunch` asserted on every WARNING captured on the Isaac
simulation logger during the test, and that logger is shared with `destroy()`:
an `IsaacSimulation` another test left to the cyclic collector is finalised at
whatever point the collector fires, and on one CI run four of them fired inside
the silent-request test and failed it on a launch that had dropped nothing. The
class now reads the report naming the launch keys it could not apply, and a new
cell plants a finalizer on the shared logger inside the window to hold it there.
