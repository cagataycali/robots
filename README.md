# action_horizon domain on the directly-drivable runner surfaces

`capture.py` drives `PolicyRunner.run` on a real headless MuJoCo `so100` for 96
control steps at three requested horizons, counting `get_actions` calls and
rendering the end state. It is run unchanged in a checkout of `main` and in the
branch; `compose.py` builds the figure and asserts every number it prints.

Measured: the honored `action_horizon=8` rollout is identical on both trees
(12 inferences, every joint equal to 6 decimals, renders agreeing to 1/255 over
32 of 471200 pixels). On `main`, `action_horizon=0` returned `status="success"`
having run **96** inferences - eight times the model calls, at a re-query
interval the caller set to 0 - and ended in a different pose (12.35% of pixels
differ). `nan` spent one inference before failing with a message naming neither
the parameter nor the method.
