### Fixed: `policies/vera` - the two sampler knobs reached the server as unchecked text

`VeraConfig` declares seven numeric fields and held five of them to a shared
domain on the *effective* value: both ports, `render_width`, `motion_plan_scale`
and `server_ready_timeout`. The two it never looked at were `sample_steps` and
`teacache_thresh`, the WAN video planner's denoise-step count and its teacache
rel_l1 threshold, so `nan`, `inf`, a negative, a zero, a `bool`, a `str` and
`None` on a field declared `float` were all accepted - 17 of 17 spellings
measured.

Neither field is read anywhere else. Their only consumer is the launch command,
which carries them as *text*: `str(cfg.sample_steps)` and
`str(cfg.teacache_thresh)` in `VeraServerRunner._build_command`, and
`VERA_SAMPLE_STEPS=` in `DockerServerRunner`'s `-e` overlay. Nothing between the
config and the server inspects the value, so the server was left to report it,
and it has two ways to - neither of which names the field.

A token the flag's own type cannot parse (`'2.7'`, `'nan'`, `'True'`, `'ten'` for
an `int` flag) makes the server exit before it opens its port, and
`_wait_until_ready` answers `VERA server exited early (code N) ... common causes
are missing checkpoints (set VERA_CKPT_ROOT / ckpt_root) or CUDA OOM` - two
causes that are not the cause. A token it *can* parse starts a server on a
setting nobody asked for: `0` or `-5` denoise steps, a threshold of `nan` (below
nothing) or `inf` (below everything), each turning the comparison the flag exists
for into a constant. Which of the two happens is not a property of the value
being usable, only of how `str()` happens to spell it. `start()` already takes
this position two statements above the launch - `_require_vera_installed` exists
because, in its own words, without it "a missing install surfaces only as an
opaque 'server exited early (code 1)' RuntimeError several seconds later".

The two spellings of each knob also disagreed. `_env_int` and `_env_float` return
`None` for anything `int()`/`float()` refuses, so `VERA_SAMPLE_STEPS=ten` is
absorbed and the planner yaml decides - deliberate, and pinned. The keyword
spelling of the same knob was checked nowhere, so one knob was guarded from the
environment and unguarded from the API.

`sample_steps` now takes the shared count domain `render_width` takes and is
converted to `int`; `teacache_thresh` takes the shared continuous domain
`motion_plan_scale` takes and is converted to `float`. The conversion is
load-bearing on the count rather than tidy: `sample_steps=20 / 2` is a positive
whole number the domain accepts, `str(10.0)` is `'10.0'`, and `--sample-steps`
cannot parse it, so a computed count ended the server before this change and
reaches the flag as `10` after it.

`0` is not the threshold's opt-out - `teacache=False` is, and it emits
`--no-teacache` in place of the flag - and the documented quality cliff above
`0.15` is guidance about output quality rather than a bound, so `0.25` stays a
legitimate request. The threshold is checked whatever `teacache` is set to,
because `VeraConfig` is a plain dataclass and the flag can be turned on after
construction.
