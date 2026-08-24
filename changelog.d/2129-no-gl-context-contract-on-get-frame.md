### Quality: pin the no-OpenGL-context contract on `get_frame`, the renderer consumer that raises

`_get_renderer` returning `None` (no EGL/OSMesa on the host) is answered
independently by each of its four consumers, and the channels deliberately
differ: `render` / `render_depth` return a `status=error` agent-tool envelope,
`_get_sim_observation` skips the camera, and `get_frame` -- which returns raw
`(rgb, depth)` ndarrays -- raises `RuntimeError`. Three of the four were pinned;
`get_frame`, the only one that raises, was not, so nothing held it to the
actionable "install EGL or OSMesa" text and nothing stopped it being
"harmonised" onto the envelope channel. That would be silently unsafe: a
two-key envelope unpacks without complaint at a `rgb, depth = sim.get_frame(...)`
call site, handing the consumer the strings `"status"` and `"content"` and
failing far from the missing GL context.

Pins the missing cell plus the two documented in-process consumers whose own
message only names GL because `get_frame` raises
(`HybridCompositor.render` propagates it; `get_world_point` carries the text
into its tool envelope), and adds a drift guard so a fifth consumer of the
shared renderer helper has to decide and record its own no-GL channel. Tests
only -- no behaviour changes, and none of the new tests needs a GL context.
