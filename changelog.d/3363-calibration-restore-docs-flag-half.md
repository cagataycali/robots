### Documentation

- **hardware/tools**: The calibration section of the tools reference documented the atomic
  restore write and closed on the report, which reads as "the store is safe now". That is
  true of a write that *fails*; it says nothing about a write the tool was told to make, and
  `overwrite` - the confirmation gate in front of the only write in `lerobot_calibrate` that
  replaces a measurement - was read by truthiness, so `overwrite="false"` selected the
  overwrite it spells the refusal of and reported `Overwrite mode: false` beside the count of
  calibrations it had just replaced. The page now carries the flag half beside the write half:
  which spellings inverted, why neither the atomic commit nor the backup recovers the
  measurement a successful overwrite replaced, where each of the two surfaces refuses, and
  that only `action="restore"` consults the flag. The existing write section points forward to
  it so a reader does not stop where it used to end.
