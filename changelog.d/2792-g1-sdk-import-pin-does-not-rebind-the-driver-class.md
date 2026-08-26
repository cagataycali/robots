### Fixed: the G1 SDK-import pin no longer rebinds the registered driver class

`test_g1_driver_module_does_not_import_unitree_sdk2py_at_load_time` pinned a
real contract - importing `strands_robots.drivers.g1` must not pull in the
vendor SDK, so the module stays importable on a stock install - by reloading
the driver module inside the running test session.

`importlib.reload` re-executes a module body into the *same* namespace, so it
rebinds every class that body defines. The driver registry captures
`G1Driver` by reference when the shipped table is registered, so after the
reload the registry's class and the class the module's own name resolves to
are two distinct objects with an identical `repr`. Every identity assertion
made later in the same session then fails, reporting
`assert <class '...G1Driver'> is <class '...G1Driver'>` - a message that names
the same class twice and points at neither the reload nor the file containing
it.

The pin now measures the import graph in a clean interpreter. Nothing in the
test session is rebound, and the contract is stated more strongly: no
`unitree_sdk2py` module is loaded at all, rather than none beyond whatever an
earlier test in the session already imported - an assertion that holds even on
a box where the SDK is installed.

A second cell reads the rebind directly, calling the pin and then comparing the
registry's class against the module's own name. Without it the file is silent:
the symptom surfaced only in `tests/drivers/test_reachy_driver.py`, whose
`test_the_g1_registration_survives_a_second_shipped_driver` compares those two
references and runs after the pin in collection order.
