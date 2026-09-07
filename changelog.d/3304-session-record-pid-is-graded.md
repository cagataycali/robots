### Fixed

- **A session record's pid is graded rather than converted.** `lerobot_teleoperate`
  and `lerobot_train` read the pid of a detached session out of a JSON store, and
  `int()` of a value that store should not hold names a different process:
  `int(4321.5)` is `4321`, so `stop` signalled - and killed - whatever held that
  number while reporting success under a number that is not a pid, and `true` is
  pid 1. The same conversion raised `ValueError` / `OverflowError` / `TypeError`
  out of reads documented to degrade to "no sessions", for values `json.load`
  produces from a well-formed file, including the U+FFFD each store's decode
  policy substitutes for a damaged byte. Both tools now read that field through
  one owner, which answers the pid or "no pid recorded", so an unusable record
  reads as stopped, is refused by name at `stop`, and is never signalled.
