### Fixed: Isaac has the introspection methods describe() advertises

`describe()` told agents the Isaac backend had `get_robot_state` and `get_features`, but calling either raised `AttributeError`. Isaac now implements both, plus `list_objects` and `list_cameras`, with the MuJoCo backend's envelopes.
