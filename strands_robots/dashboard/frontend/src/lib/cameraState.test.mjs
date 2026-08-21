import assert from 'node:assert/strict'
import { classifyCamera, ageText, publishedAtMs, PUBLISH_FRESH_MS, CAPTURE_STALE_MS, STALL_MS } from '/tmp/cameraState.mjs'

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

// --- ARRIVAL IS NOT CAPTURE (found on the live page, iteration 41) ----------
// The camera socket replays the peer's last cached frame to a new subscriber, so
// a frame taken this morning ARRIVES now. Measured on so101-arm-1's wrist tile:
// class "cam-stalled frozen", text "last frame 8s ago", over pixels the peer had
// captured 24,307 seconds (6.8h) earlier - and for the first 2.5s after connect
// the same tile was classified `live`, at full brightness.
const replayed = classifyCamera({
  now: NOW, conn: 'open', frames: 3, lastFrameAt: NOW - 1000,   // arrived 1s ago
  publishedAt: (NOW - 24_307_000) / 1000,                        // captured 6.8h ago
})
assert.notEqual(replayed.kind, 'live', 'a replayed cache entry must never read as live')
assert.equal(replayed.kind, 'stalled')
assert.equal(replayed.live, false)
assert.equal(replayed.frozen, true, 'the pixels are old: the image must be dimmed')
assert.equal(replayed.title, 'stale frame')
assert.match(replayed.detail, /captured this 6\.8h ago/)
assert.match(replayed.detail, /arrived here 1s ago/)
assert.match(replayed.detail, /replay of its last frame, not a new one/)
// The capture age is ATTRIBUTED, never asserted: it is computed across two
// machines' clocks, and a phone minutes behind must not become a verdict about
// the hardware.
assert.match(replayed.detail, /the peer says/)

// A fresh capture that arrives fresh is still simply live - the new rule must not
// make every camera look broken.
const reallyLive = classifyCamera({
  now: NOW, conn: 'open', frames: 30, lastFrameAt: NOW - 200, publishedAt: (NOW - 200) / 1000,
})
assert.equal(reallyLive.kind, 'live')
assert.equal(reallyLive.live, true)

// Both clocks stale: the arrival age leads (the socket really did stop) and the
// capture age is added, because the two together say "the camera stopped, and so
// did the stream" rather than "someone reconnected to a corpse".
const bothStale = classifyCamera({
  now: NOW, conn: 'open', frames: 8, lastFrameAt: NOW - 60_000, publishedAt: (NOW - 90_000) / 1000,
})
assert.equal(bothStale.title, 'stalled')
assert.match(bothStale.detail, /last frame 60s ago/)
assert.match(bothStale.detail, /captured it 2m ago/)  // 90s crosses into minutes by design

// A peer clock AHEAD of ours can only be skew. It is discarded, not read as
// freshness - and it must not manufacture staleness either.
const clockAhead = classifyCamera({
  now: NOW, conn: 'open', frames: 4, lastFrameAt: NOW - 300, publishedAt: (NOW + 600_000) / 1000,
})
assert.equal(clockAhead.kind, 'live', 'negative capture age is skew, not evidence')
assert.doesNotMatch(clockAhead.detail, /captured/)

// No capture timestamp at all: fall back to arrival exactly as before. Absence
// of a clock is not evidence of an age.
const noCapture = classifyCamera({ now: NOW, conn: 'open', frames: 4, lastFrameAt: NOW - 9000 })
assert.equal(noCapture.title, 'stalled')
assert.equal(noCapture.detail, 'last frame 9s ago')

// The threshold is a threshold, and injectable.
assert.equal(classifyCamera({
  now: NOW, conn: 'open', frames: 4, lastFrameAt: NOW - 100,
  publishedAt: (NOW - (CAPTURE_STALE_MS - 2000)) / 1000,
}).kind, 'live')
assert.equal(classifyCamera({
  now: NOW, conn: 'open', frames: 4, lastFrameAt: NOW - 100,
  publishedAt: (NOW - (CAPTURE_STALE_MS + 2000)) / 1000,
}).kind, 'stalled')
assert.equal(classifyCamera({
  now: NOW, conn: 'open', frames: 4, lastFrameAt: NOW - 100,
  publishedAt: (NOW - 3000) / 1000, captureStaleMs: 1000,
}).kind, 'stalled', 'captureStaleMs must be honoured')

// An ERROR still outranks both clocks: the reason beats the timing.
const errFirst = classifyCamera({
  now: NOW, conn: 'open', frames: 4, lastFrameAt: NOW - 100,
  publishedAt: (NOW - 24_307_000) / 1000, error: 'camera in use',
})
assert.equal(errFirst.kind, 'busy')
assert.equal(errFirst.frozen, true, 'old pixels are still on screen and still marked')

console.log('cameraState: 62 assertions ok')

// --- the stall threshold is asserted against the REAL constant, not a copy of it -------
// STALL_MS was exported for exactly this and no test read it, so the number here and the
// number in the product were free to drift apart in silence. It governs a camera that HAS
// delivered frames and then stopped — the frozen-tile case, not a never-opened one.
{
  const cam = (sinceLast) => classifyCamera({
    now: NOW, conn: true, frames: 5, lastFrameAt: NOW - sinceLast,
  })
  assert.notEqual(cam(STALL_MS - 200).kind, cam(STALL_MS + 200).kind,
    'the verdict must change across the real threshold')
  assert.equal(cam(STALL_MS + 200).frozen, true, 'past the threshold the tile is frozen, not live')
}
