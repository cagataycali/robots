### Changed: the dashboard's sim peer tool is the simulation tool

A simulation peer's tool on the dashboard offered eight mesh verbs, so "add a
red cube into that sim" had no tool to call. The proxy now offers every
published simulation action the wire carries over `sim_call` (add_object,
list_objects, move_object, add_camera, render, get_robot_state, move_to,
set_gripper and the rest) with the published parameters as fields, refuses a
denied action or a peer host path before any round trip with the wire's reason,
and hands a render back to the model as an image. `spawn_robot` registers the
new peer's tools into the live agent before it returns, so the same message
that creates a robot can put something into its world; the fleet signature and
the motion hook follow. A mesh session that refuses to open (a plain start on a
laptop, where auth defaults to mtls with no certificate) now leaves its reason
on the bridge, `/api/fleet` and the mesh snapshot carry it, and the page shows
a banner naming it with the `STRANDS_MESH_LOCAL_DEV=1` line that fixes it on a
trusted machine.
