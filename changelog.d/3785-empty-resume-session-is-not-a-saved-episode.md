### Fixed: a resumed recording session that captured nothing is refused, not reported as a saved episode

`start_recording` says when it resumes an existing dataset (and how to start
from scratch), and `stop_recording` measures the session against it: nothing
appended is refused naming the unchanged dataset, and a real append reports
`(+N frames, +M episode(s) this session)` beside the dataset totals. One shared
seam opens the session, so the MuJoCo, Newton and Isaac backends all report the
session rather than the dataset they resumed.
