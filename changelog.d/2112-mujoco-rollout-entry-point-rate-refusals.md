### Quality: pin the recording-rate refusal on the two MuJoCo-only rollout entry points

Seven surfaces check that a rollout's ``control_frequency`` matches the rate an
open recording declares. Five had returned the refusal at least once in the test
suite; ``start_policy`` and ``run_multi_policy`` - the two entry points that
exist only on the MuJoCo backend - never had. Both were covered by a structural
sweep asserting the guard is *called*, which cannot distinguish that from a guard
whose refusal is discarded: dropping the ``return err`` while keeping the call
leaves all 79 pre-existing cases in the module green.

The sweep gave its reason for staying structural as a driver possibly needing "a
checkpoint, a benchmark registration or a live background thread to reach".
Neither of these needs one - the rate guard sits above ``self._executor.submit``,
so ``start_policy`` returns the refusal on the caller's own thread with no worker
in existence, which is exactly what its own comment says the placement is for
("a refusal after submit would report 'started' to a caller whose rollout cannot
be recorded correctly").

Fifteen behavioural cases now pin, for both entry points: the library defaults
are refused, the message names both rates, the distortion factor and both
remedies, the envelope is the shared helper's verbatim, no frame reaches the
recorder or the dataset, and - for the async path - no worker is submitted and
the robot is not marked running. Controls pin that an aligned rate still starts
and still records, and that no recording open is never refused. No library
behaviour changes.
