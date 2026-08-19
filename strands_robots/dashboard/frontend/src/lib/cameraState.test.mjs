import assert from 'node:assert/strict'
import { classifyCamera, ageText, publishedAtMs, PUBLISH_FRESH_MS } from '/tmp/cameraState.mjs'

const NOW = 1_787_180_409_000 // ms, matching the live snapshot this was measured on

// --- the defect: `publishedAt` was read as a BOOLEAN -------------------------
// Live measurement on so101-arm-1: `top` published at 1787180409 (now) while
// `wrist` sat at 1787156394 - 6.7 HOURS old - and both were rendered by the same
// branch, saying "peer published frames, none arrived yet". "Yet" is a promise.

const stale = classifyCamera({
  now: NOW, conn: 'open', frames: 0, publishedAt: 1_787_156_394,
})
assert.equal(stale.kind, 'silent', 'a camera dead for hours must not read as "waiting"')
assert.doesNotMatch(stale.detail, /yet/, 'no "yet": nothing is on its way')
assert.match(stale.detail, /6\.7h/, `the age itself must be in the sentence: ${stale.detail}`)
// It also says WHERE it stopped, so the next step is that robot's log and not a
// page reload: the socket is healthy, the camera at the far end is not.
assert.match(stale.detail, /stopped there, not in transit/)
assert.equal(stale.live, false)
assert.equal(stale.frozen, false)

// A peer that published a moment ago IS genuinely warming up - and now says how
// long ago, which is the difference between a wait and a hang.
const warming = classifyCamera({
  now: NOW, conn: 'open', frames: 0, publishedAt: (NOW - 3000) / 1000,
})
assert.equal(warming.kind, 'waiting')
assert.match(warming.detail, /3s ago/)
assert.match(warming.detail, /none arrived here yet/)

// The boundary is a threshold, not a vibe, and it is injectable for tests.
const justInside = classifyCamera({
  now: NOW, conn: 'open', frames: 0, publishedAt: (NOW - (PUBLISH_FRESH_MS - 1000)) / 1000,
})
assert.equal(justInside.kind, 'waiting')
const justOutside = classifyCamera({
  now: NOW, conn: 'open', frames: 0, publishedAt: (NOW - (PUBLISH_FRESH_MS + 1000)) / 1000,
})
assert.equal(justOutside.kind, 'silent')
assert.equal(
  classifyCamera({ now: NOW, conn: 'open', frames: 0, publishedAt: (NOW - 5000) / 1000, publishFreshMs: 1000 }).kind,
  'silent', 'publishFreshMs must be honoured so a slow link can widen the window',
)

// No publish time at all is still the honest "sending nothing" - absence of a
// timestamp is not evidence of an age, and inventing one would be the same lie
// in the other direction.
const noPub = classifyCamera({ now: NOW, conn: 'open', frames: 0 })
assert.equal(noPub.kind, 'silent')
assert.match(noPub.detail, /stream open, camera sending nothing/)
for (const bad of [0, -1, NaN, undefined, null]) {
  const got = classifyCamera({ now: NOW, conn: 'open', frames: 0, publishedAt: bad })
  assert.equal(got.kind, 'silent', `publishedAt=${bad} must not fabricate an age`)
  assert.match(got.detail, /sending nothing/)
}

// --- units: python writes SECONDS, Date.now() is MILLISECONDS ---------------
// The dangerous half of this fix. Comparing the two directly turns 2026 into
// 1970 and every age into ~57 years, which would have read as a confident
// "the peer's last frame is 20871d old" on a perfectly live camera.
assert.equal(publishedAtMs(1_787_156_394), 1_787_156_394_000)
assert.equal(publishedAtMs(1_787_156_394_000), 1_787_156_394_000, 'ms input must pass through')
assert.equal(publishedAtMs(undefined), undefined)
assert.equal(publishedAtMs(0), undefined)
assert.equal(publishedAtMs(-5), undefined)
assert.equal(publishedAtMs(NaN), undefined)
const inSeconds = classifyCamera({ now: NOW, conn: 'open', frames: 0, publishedAt: (NOW - 4000) / 1000 })
const inMillis = classifyCamera({ now: NOW, conn: 'open', frames: 0, publishedAt: NOW - 4000 })
assert.equal(inSeconds.detail, inMillis.detail, 'seconds and ms for the same instant must agree')

// --- ageText: a duration a human reads at a glance --------------------------
assert.equal(ageText(400), '<1s')
assert.equal(ageText(4200), '4s')
assert.equal(ageText(89_000), '89s')
assert.equal(ageText(120_000), '2m')
assert.equal(ageText(24_015_000), '6.7h')   // the measured wrist camera
assert.equal(ageText(3_600_000), '60m')     // still minutes just under the hour cut
assert.equal(ageText(7_200_000), '2h')      // no trailing ".0"
assert.equal(ageText(259_200_000), '3d')
assert.equal(ageText(-1), 'unknown')
assert.equal(ageText(Infinity), 'unknown')
for (const ms of [400, 4200, 89_000, 120_000, 24_015_000, 7_200_000]) {
  assert.doesNotMatch(ageText(ms), /NaN|undefined/)
}

// --- a long STALL now reads in the right unit too ---------------------------
// Same bug, milder: a tile frozen for two hours said "last frame 7200s ago".
const longStall = classifyCamera({
  now: NOW, conn: 'open', frames: 12, lastFrameAt: NOW - 7_200_000,
})
assert.equal(longStall.kind, 'stalled')
assert.equal(longStall.frozen, true, 'the pixels on screen must still be marked stale')
assert.match(longStall.detail, /last frame 2h ago/)

// --- nothing else moved -----------------------------------------------------
const live = classifyCamera({ now: NOW, conn: 'open', frames: 5, lastFrameAt: NOW - 100 })
assert.equal(live.kind, 'live')
assert.equal(live.live, true)
const denied = classifyCamera({ now: NOW, conn: 'closed', frames: 0, error: 'TCC: not authorized' })
assert.equal(denied.kind, 'unauthorized')
const busy = classifyCamera({ now: NOW, conn: 'open', frames: 0, error: 'camera in use' })
assert.equal(busy.kind, 'busy')
// A stall outranks a fresh publish claim: the frames we SAW decide what is on
// screen, not what the peer says it is doing.
const stallBeatsPublish = classifyCamera({
  now: NOW, conn: 'open', frames: 9, lastFrameAt: NOW - 9000, publishedAt: NOW / 1000,
})
assert.equal(stallBeatsPublish.kind, 'stalled')

console.log('cameraState: 40 assertions ok')
