### Bug Fixes

- **tools**: A calibration write that cannot finish leaves the measurement already on
  disk readable. Both writers of the calibration store encoded straight into the
  destination - `save_calibration` through `json.dump`, the `"restore"` action through
  `shutil.copy2` - which truncates the stored file before the first byte of the
  replacement lands, so a value the encoder rejects or a filesystem that refuses the
  write left a prefix where the measurement was. Nothing downstream reported it:
  `calibration_exists` answers `True` for a prefix, `load_calibration` answers `None`,
  and `action="view"` renders the path, size and timestamp under `status="success"` with
  no motors in it. `"restore"` was the sharper of the two, being the path a lost
  calibration is recovered on - with `overwrite=True` a refused write destroyed the
  calibration being replaced *and* did not install the backup. The calibration is now
  serialized in full before the stored file is touched and committed through a temp file
  plus `os.replace`, in one owner both writers ask through, so a failed write leaves the
  recorded homing offsets and joint travel limits byte-identical and still loadable.
