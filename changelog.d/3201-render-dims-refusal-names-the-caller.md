### Fixed: a render-dimension refusal names the method that was called

Both the MuJoCo and the Newton backend funnel the width/height domain of every
render surface through one helper -- `_validate_render_dims` on MuJoCo,
`_resolve_camera_view` on Newton -- and each helper spelled the subject of its
own refusals as the literal `render`. Each serves more than `render`, so five of
the eleven entry points across the three backends reported a method the caller
had not called. Measured with `width=0` against a live world on each backend:

| backend | called | before | after |
|---|---|---|---|
| mujoco | `render` | `render: ...` | unchanged |
| mujoco | `render_depth` | **`render: ...`** | `render_depth: ...` |
| mujoco | `get_frame` | **`render: ...`** | `get_frame: ...` |
| mujoco | `get_camera_params` | **`render: ...`** | `get_camera_params: ...` |
| mujoco | `add_camera` | `add_camera: ...` | unchanged |
| newton | `render` | `render: ...` | unchanged |
| newton | `get_frame` | **`render: ...`** | `get_frame: ...` |
| newton | `get_camera_params` | **`render: ...`** | `get_camera_params: ...` |
| isaac | `get_frame` / `get_camera_params` | names itself | unchanged |

The reader is usually an agent -- `render` and `add_camera` are tool-callable and
return the text in an error envelope, while `get_frame` and `get_camera_params`
raise it -- so the repair the message suggests is a call the caller never made.
`render` also has a different signature and return type from the two raising
methods, so following it means editing an unrelated call site.

That the misattribution mattered was already established in the tree twice over.
MuJoCo's `add_camera` repaired it after the fact with
`text.replace("render:", "add_camera:", 1)`, a coupling to the literal prefix of
all four messages the guard can return which a rewording would silently break;
and Isaac never had the defect, passing its own name to `positive_count_error` at
each call site. The ~20 domain helpers in `strands_robots.utils` all take their
caller's name for the same reason -- these two guards were the exception.

Both now take a required `context`, so each entry point names itself and the
string patch is gone. Required rather than defaulted to `"render"`, because a
default is the shape that produced the defect. Isaac is unchanged: it is the
control.

Two existing parity suites asserted the two texts were byte-identical, which is
a stronger claim than the drift they exist to catch -- byte-identity also
*required* `render_depth` and `get_camera_params` to report `render`, so the
misattribution was the pinned behaviour. Both now compare the reason with each
subject stripped, keeping the anti-drift guarantee while making a wrong subject a
failure. The new suite derives the expected subject from each call's own
enclosing method, so a sixth entry point cannot join either funnel without naming
itself -- the failure mode that produced this.
