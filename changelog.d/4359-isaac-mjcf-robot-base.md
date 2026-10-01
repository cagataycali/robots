### Fixed: an MJCF robot on Isaac spawns where it was asked, as a fixed-base arm, without a second floor

Three defects in how a robot converted from MJCF was put on the stage, every
registry robot included:

- `add_robot(position=...)` was ignored: the position was written to the PhysX
  view only, the robot's world weld stayed anchored at the origin and pulled
  it back, and `World.reset()` re-read USD. Two robots therefore overlapped
  and dragged each other (so100: 1.45 rad of cross-talk). The position is now
  authored on the robot's container prim, which the weld is expressed
  against, so it survives resets; a floating go2 asked for z=0.3 lands at
  0.741 m (MuJoCo 0.745).
- The articulation root sat on the base rigid body, welded by a separate joint
  - a floating-base articulation to PhysX. `get_jacobian` refused every arm
  ("Only fixed-base articulations are supported"), the delta-EEF controller
  failed on every action, and `set_robot_pose` was undone within a step or
  drove the arm to NaN. The root now moves to the weld, as it already did for
  URDF robots.
- The menagerie `scene.xml` floor was imported inside each robot, one extra
  infinite ground per robot. It is deactivated at load.

`get_body_state("<robot>/<link>")` and `gripper_frame_pose` also searched the
whole `/World` and answered the FIRST robot's link for every robot; they now
look under the named robot first.
