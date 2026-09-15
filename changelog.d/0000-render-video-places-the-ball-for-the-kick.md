### Fixed: `render_video.py` places the ball where the kick weights were trained to find it

`scene_ball.xml` declares its ball 0.3 m straight ahead; `ball_kick_left` and
`ball_kick_right` were trained with it 0.09 m ahead and 0.042 m to the side of
the kicking foot, in the robot's yaw frame - which `docs/policies/microduck.md`
already says, while the example's own recipe (`--onnx ball_kick_left.onnx
--scene scene_ball.xml`) rendered four seconds of the duck kicking air (min
robot-geom distance to the ball 0.252 m, no contact, ball travel 0 m on the L40S).

The example now teleports the ball to the trained offset before the rollout,
the way Pollen's runtime does before every kick - in front of the foot
`--kick-foot left|right` names, inferred from the weight's file name when
omitted - and says where it put it. Same rollout after the change: contact,
ball travel 0.483 m. A scene without a ball is left alone; `--kick-foot` on
such a scene is refused and points at `--scene scene_ball.xml`.
