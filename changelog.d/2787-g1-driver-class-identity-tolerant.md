### Test: the G1-registration-survives assertion tolerates a second class object under a re-import

`test_the_g1_registration_survives_a_second_shipped_driver` in
`tests/drivers/test_reachy_driver.py` graded the registration by
`get_native_driver_class("unitree_g1") is G1Driver`. Under some test orderings
the `_register_shipped_drivers` loop reaches `strands_robots.drivers.g1` a
second time - either via a test that reloads the module or via a stale
`sys.modules` entry from an earlier suite phase - and the second import
creates a fresh `G1Driver` class *object* with the same fully-qualified name.
Both are the same registration for the registry's purposes (the qualname the
seam looks up hasn't moved); the `is` check reports otherwise on the class
object only, and the resulting failure is a false negative: the registration
did survive, `is` just cannot tell.

Fix: compare `__module__` and `__qualname__` instead of the class object. That
is what a caller-facing consumer of the registry actually reads (drivers are
looked up by module path, not by class object identity - `register_native_driver`
stores by canonical robot name, not by class), and the pattern lerobot's own
plugin registration uses.

Observed on #2784's `call-test-lint` run on head `99efd10a` at
`2026-08-26T17:36:18Z`:

```
tests/drivers/test_reachy_driver.py:722: AssertionError
assert <class 'strands_robots.drivers.g1.G1Driver'> is <class 'strands_robots.drivers.g1.G1Driver'>
```

The two classes have the same repr because they *are* the same registration
under a different identity. #2762 (which added
`tests/drivers/test_reachy_driver.py`) shipped fine when it was the only
driver in the suite; the flake surfaces when it sits alongside another
driver-registering test that ordering-permutes ahead of it. This entry keeps
the test's *intent* (registration must survive) without demanding a
brittle-under-reimport identity check.
