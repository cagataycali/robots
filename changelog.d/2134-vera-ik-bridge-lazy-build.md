### Quality: pin the lazy `MinkIKBridge` build that every eef-delta rollout takes once

`VeraPolicy._ensure_ik_bridge` builds the IK bridge on demand and caches it on the
policy. Three sibling contracts pin what happens around that build - that
`set_ik_target` clears it "so it rebuilds", that a refused `set_ik_target` does
not, and that a later call reuses the one already there - but all three inject a
bridge (`policy._ik_bridge = FakeBridge()`), so `if self._ik_bridge is None` was
always False and the two statements that construct one had never run.

That left the build unpinned in the way that matters: nothing asserted the
rebuild the first message names, or the frame the build is handed. Measured
against the pre-existing 728 vera tests, a build that swapped the frame name and
type arguments, hardcoded `"body"` over the caller's choice, or returned a fresh
bridge without caching is invisible to all of them.

These tests drive the build through the public `get_actions` path against a real
compiled model - a first inference, where every eef-delta rollout goes - and
assert on the bridge it produced rather than on one handed to it: that the frame
name and type reach it, that a later inference reuses it, and that retargeting
mid-session rebuilds against the new frame. `provider.py` goes from 98% to 99%
with the build's lines covered. No production code changes.
