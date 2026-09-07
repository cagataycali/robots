### Bug Fixes

- **tools/lerobot_camera**: `capture_batch` read its `camera_ids` selector by
  truthiness, so `[]` - the selection a filter that matched nothing produces -
  was widened to the two default robot cameras and reported as "2/2 cameras"
  under `status="success"`, a single id passed as a bare string was read one
  camera per character (eleven one-character cameras for `"/dev/video4"`), a
  mapping was iterated over its keys and a repeated id opened one device twice.
  The selector is now read `is None`: omitting it still selects the defaults,
  and every other shape is refused by value before the save directory, the
  thread pool or any camera exists. A well-formed list is unchanged.
