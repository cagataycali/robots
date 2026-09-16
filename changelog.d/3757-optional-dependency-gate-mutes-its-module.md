### Fixed: an optional-dependency gate no longer skips the whole module it sits in

`pytest.importorskip` raises `Skipped` rather than returning `None`, and a
decorator is evaluated while the module is imported, so a gate written as
`@pytest.mark.skipif(pytest.importorskip("x") is None, ...)` skipped the entire
file instead of the test it named. Two gates in the suite were doing this: on an
interpreter without the extra, `tests/policies/test_a_misspelled_provider_kwarg_is_refused_naming_the_one_meant.py`
collected nothing instead of 17 cells, and
`tests/simulation/test_sim_engine_describe_discovery.py` collected nothing
instead of 33 -- 50 cells needing neither `zmq` nor `mujoco`, dropped by gates
protecting the 18 that do. Both now read presence as a spec
(`importlib.util.find_spec`), the convention the rest of the suite keeps, and
`tests/test_an_optional_dependency_gate_does_not_mute_its_module.py` refuses the
shape tree-wide.
