### Fixed: `add_camera(parent_body=...)` without a pose is refused instead of mounting the camera 1.73 m off the body

A camera mounted on a body reads `position` and `target` in that body's frame;
omitting both used to apply the free camera's world-frame defaults and return
success with a "wrist" camera looking back at the arm from 1.73 m away. Both
backends now refuse, naming the frame and - when the body carries the robot's
end-effector site - a starting pose that frames the fingertips. `list_bodies`
and the camera-naming guide name the pose arguments.
