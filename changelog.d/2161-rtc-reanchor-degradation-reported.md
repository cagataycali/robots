### Fixed: a relative-action RTC leftover that cannot be re-anchored now says so

For a relative-action flow policy (pi0 / pi0.5 / pi0-FAST), the unexecuted tail
of a chunk is only valid in the coordinate frame of the observation that
produced it. `LerobotLocalPolicy` therefore keeps that tail in absolute robot
coordinates so the next chunk can be re-expressed against the moved state;
`_absolute_rtc_leftover` performs the conversion and returns `None` when it
cannot. Three conditions produce that `None` and only one is benign - an
absolute-action policy has no frame shift to undo. The other two, a bridge with
no postprocessor and a postprocessor that does not yield a plain action tensor,
left a relative-action policy carrying a stale-frame prefix into every
subsequent chunk with no signal at all, behind `_resolve_rtc_rebase_steps`'
INFO line announcing that re-anchoring was enabled.

That is the same consequence `_resolve_rtc_rebase_steps` already warns about
when LeRobot's re-anchor helper is unavailable ("the chunk-seam prefix will be
carried in a STALE coordinate frame"), and which its own regression test pins as
"warn once ... never crash or silently drop the prefix". Both degradations now
report once per policy in that wording, naming the cause alongside the effect,
using the module's existing one-shot warn latch. The benign absolute-action case
stays silent, and a postprocessor that *raises* stays fatal rather than being
downgraded to a silent frame shift. The method's docstring previously named two
conditions for three exits; it now names all three and which are degradations.
