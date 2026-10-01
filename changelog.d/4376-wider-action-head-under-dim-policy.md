### Fixed: `lerobot_local` embodiments drive a wider action head under `dim_policy` pad/truncate, and the width is judged before the download

`EmbodimentMap.validate` refused any action-width mismatch whatever
`dim_policy` said, so `embodiment=` could not drive `lerobot/pi0_base` or
`pi05_base` (a 32-wide padded head for every embodiment), and refused only
after the two-minute load. `EmbodimentMap.action_dim_error` is now the one
rule: `strict` wants the exact width; `pad` / `truncate` accept a wider head
whose leading columns drive the actuators and still refuse a narrower one.
`validate` reads it after the load and `preflight` reads it before the
download from the width the checkpoint's `config.json` declares
(`declared_action_dim`), so a too-narrow head is a `status=error` envelope
before any weights move. (#4193)
