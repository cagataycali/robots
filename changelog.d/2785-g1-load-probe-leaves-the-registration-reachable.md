### Fixed: probing the G1 driver's load-time imports no longer orphans its registration

`tests/drivers/test_g1_driver.py` pins that importing
`strands_robots.drivers.g1` pulls in no `unitree_sdk2py` module, so the driver
stays importable on Thor and CI without the SDK. Observing that needs a genuine
re-execution - the module is already imported by the time any test runs - and the
cell used `importlib.reload`.

`reload` re-executes the source into the *live* module object, rebinding every
attribute it defines. `strands_robots.drivers.g1.G1Driver` therefore became a
different class object from the one `strands_robots.drivers` registered from
`_SHIPPED_DRIVERS` at first import, and nothing put the original back:
`get_native_driver_class("unitree_g1")` answered with a class no importer could
reach for the remainder of the session. That is the orphan AGENTS.md > Testing
Patterns forbids for a `sys.modules` removal, reached through `reload` instead,
and it carried the same cost the rule records - a double installed on the module
attribute is not read through the registered class.

It surfaced as a cross-file failure with no local cause. `test_reachy_driver.py`'s
`test_the_g1_registration_survives_a_second_shipped_driver` compares the
registered class to the imported one by identity, and read as
`assert <class 'strands_robots.drivers.g1.G1Driver'> is <class
'strands_robots.drivers.g1.G1Driver'>` - two identically named classes, on a
branch touching neither file. Any cell that reads the registration that way would
have done; this one only failed because `tests/drivers/test_g1_driver.py` sorts
ahead of it.

The probe now executes the same source into a throwaway module object, which
observes exactly the same module-scope imports and leaves both the live module and
the registration untouched. That is safe because the driver imports only absolute
paths and does not register itself - the registration loop lives in the package
`__init__` - so a second execution has no side effect to undo. The identity
assertion that caught the orphan is left as it was, since it was reporting
correctly.
