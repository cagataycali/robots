### Fixed: `lerobot_async` sends only the cameras the checkpoint declares

A lerobot `PolicyServer` resizes every declared `observation.images.<key>` by
the checkpoint's own `policy_image_features`, so a camera the checkpoint does
not declare is a `KeyError` inside `prepare_raw_observation` - which the server
reports only as an empty action chunk, and the client only as "server returned
no actions". `LerobotAsyncPolicy` declared every camera array present in the
observation, so a robot exposing more cameras than the checkpoint could not
reach the server at all: a MuJoCo world always carries its implicit `default`
free camera, and a stock `lerobot/smolvla_base` (which declares `camera1`,
`camera2`, `camera3`) refused every observation from one.

`image_keys=[...]` now scopes the declaration and the wire to an ordered subset
of the robot's cameras - the async analog of `lerobot_local`'s knob of the same
name, held to the shared `name_list_error` domain, composing with `rename_map`,
and refusing a named camera the observation does not carry. Omitted, it still
declares every camera, so a robot whose cameras are exactly the checkpoint's is
unchanged. The empty-chunk refusal now also names the cameras it sent and the
knob that scopes them, instead of pointing only at the server log.
