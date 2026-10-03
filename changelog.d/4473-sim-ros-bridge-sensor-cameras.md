### Fixed: the sim ROS 2 bridge no longer publishes the free overview camera

With `ros2_bridge=True` a simulation advertised `/<robot>/default/image_raw` for
the implicit `default` overview camera, a view no real arm carries, so the sim
and the robot it mirrors differed on the ROS 2 graph. `SimRosBridge` now
publishes only the cameras added to the scene; Foxglove is unchanged.
