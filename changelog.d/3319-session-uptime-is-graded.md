### Bug Fixes

- **tools**: a session record's `start_time` is now read by one owner,
  `_process_stop.session_uptime`, instead of being subtracted from the clock at
  four call sites in three different spellings. A record that stated no usable
  start reported a duration the session never had - an absent `start_time`
  defaulted to `0` in the arithmetic and rendered as `Uptime: 29813993.5 min`, a
  little under fifty-seven years, under `status: success` and beside a correct
  running flag; `NaN` rendered `nan min` and a stamp ahead of this clock rendered
  a negative duration. `null`, a string, a list and a mapping raised out of the
  `list` verb, which then reported none of the sessions the store holds, so
  running sessions became invisible because one unrelated record was damaged. The
  reported `Uptime` field now says which way a record did not state its start,
  and the `uptime` seconds in the `json` block is `None` rather than a span the
  reader could not measure. A real stamp renders exactly as before.
