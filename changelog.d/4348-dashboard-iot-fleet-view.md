### Added: a robot reached over AWS IoT Core is a full dashboard card, next to the LAN peers

On the `bridge` backend every fleet card carries a `reach` chip (`lan`, `iot`,
`both`), and a robot heard over IoT shows what a LAN peer shows: joints and
state, the sensor strips, safety state, its advertised tool, run / stop / reset
through the same gates, and its cameras. A camera S3 reference
(`camera/<cam>/ref` from `camera_offload`) is resolved by the dashboard itself,
server side, from an `https` URL on an `amazonaws.com` host within 3 s and 8 MB,
one fetch in flight per camera; the presigned URL never reaches the browser.
`STRANDS_MESH_IOT_CAMERA_INLINE=1` on a robot lets it publish JPEG frames under
the 128 KB AWS payload cap instead (over the cap: dropped, one warning per topic).
Every tile shows where the frame came from and the publisher to dashboard
latency. A provisioned Thing that never spoke is a grey registry card with a
`ping` button: one `status` read, point to point, answered by the Thing or by the
broker's 404. Three fixes underneath: the `strands-operator` policy could not
receive most of what the dashboard subscribes to (the new grants ride a second
policy, `strands-operator-observe`, because AWS caps one document at 2048
characters; `_ensure_policy` now dumps compact and refuses an oversize document,
and a grader pins every shipped document under the cap); and the dashboard's
robot-less safety Mesh announced itself as `<dashboard>-safety`, a name the
operator policy does not let it publish under, so the first command from the
dashboard dropped its IoT session: on `iot` and `bridge` the rail is now the
Thing. Existing fleets run `provision_operator` or `reprovision_thing` once for
one operator to get the second policy.
