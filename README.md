# RTC re-anchoring degradation report

`capture.py` drives `LerobotLocalPolicy._predict_with_rtc` twice with the robot
state moved between chunks, once per fallback, recording the prefix actually
handed to the denoiser and every log record. It is run unchanged in a worktree
at `upstream/main` and on the branch; `compose.py` asserts every rendered value
against the two JSON dumps before drawing.
