### Fixed: Isaac move_to aims the registry's tool frame, like MuJoCo

MuJoCo's `move_to` solves for a robot's registry `tool_frame` site (so100: the point between the jaws); Isaac's IK model never carried it and solved for the wrist body. Isaac now adds the same site, so both backends move the same point.
