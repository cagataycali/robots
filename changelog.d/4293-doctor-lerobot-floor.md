### Fixed: the doctor fails a lerobot below 0.6.1 instead of printing PASS

pip keeps a pre-existing older lerobot when `strands-robots` is installed, and the doctor printed `PASS  lerobot 0.5.1` for it; the first `stream_dataset()` then raised `TypeError` about `return_uint8`, a keyword the caller never passed. The LeRobot row now fails below 0.6.1 with `pip install -U 'strands-robots[lerobot]'` as the fix, and the streaming path raises `ImportError` naming the installed version and the same remedy.
