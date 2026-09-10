### Fixed: the recorder warm-up tests record into `tmp_path`, not `/tmp`

Three `test_rendering.py` warm-up tests passed `output_dir="/tmp"`. Since #930
`start_cameras_recording` refuses a symlinked output directory, and on macOS
`/tmp` is a symlink to `/private/tmp`, so the three failed on every Mac
("output_dir '/tmp' is a symlink - refusing to follow") while CI on Linux
never saw it. They now use pytest's `tmp_path`.
