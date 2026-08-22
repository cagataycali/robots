// Assertions for the record screen's "are these numbers live?" verdict
// (lib/sessionFreshness.ts). Run: npx esbuild src/lib/sessionFreshness.ts --bundle
// --format=esm --outfile=/tmp/sessionFreshness.mjs \ && node src/lib/sessionFreshness.test.mjs
import assert from 'node:assert/strict'

const { sessionFreshness, staleSuffix, STALE_AFTER_MS } =
  await import('/tmp/sessionFreshness.mjs')

const NOW = 1_800_000_000_000

// A single lost tick IS a blip: the poller retries at 1Hz and a banner every
// time a packet slips would train the operator to ignore the banner.
{
  const f = sessionFreshness({ lastOkAtMs: NOW - 1500, nowMs: NOW, recording: true })
  assert.equal(f.stale, false)
  assert.equal(f.text, null)
}
assert.ok(STALE_AFTER_MS > 2000 && STALE_AFTER_MS < 6000, 'tolerate a couple of ticks, not ten')

// THE BUG: the frame counter frozen while presented as live. During a take that
// number is the only evidence data is being captured.
{
  const f = sessionFreshness({ lastOkAtMs: NOW - 9000, nowMs: NOW, recording: true, lastError: 'cannot reach robots.cagatay.my' })
  assert.equal(f.stale, true)
  assert.equal(f.ageS, 9)
  assert.equal(f.tone, 'bad', 'a freeze mid-take is the dangerous one')
  assert.match(f.text, /frame counter is NOT updating/)
  assert.match(f.text, /last session read 9s ago/)
  assert.match(f.text, /cannot reach robots\.cagatay\.my/, 'the reason travels with the verdict')
  // The three worlds a frozen counter hides are named instead of guessed.
  assert.match(f.text, /may still be recording, or the episode may already have ended/)
  assert.match(f.text, /this page cannot tell/)
  assert.match(f.text, /not live/)
}

// At rest the same freeze is real but not urgent — and must not claim the
// buttons are broken.
{
  const f = sessionFreshness({ lastOkAtMs: NOW - 5000, nowMs: NOW, recording: false })
  assert.equal(f.stale, true)
  assert.equal(f.tone, 'warn')
  assert.match(f.text, /session state is 5s old/)
  assert.match(f.text, /Buttons still work/)
  assert.doesNotMatch(f.text, /frame counter/)
}

// Recording vs at rest are different sentences.
assert.notEqual(
  sessionFreshness({ lastOkAtMs: NOW - 5000, nowMs: NOW, recording: true }).text,
  sessionFreshness({ lastOkAtMs: NOW - 5000, nowMs: NOW, recording: false }).text,
)

// No successful read yet: the initial load owns that message, so stay quiet
// rather than stacking a second banner.
for (const bad of [null, undefined, NaN]) {
  const f = sessionFreshness({ lastOkAtMs: bad, nowMs: NOW, recording: true })
  assert.equal(f.stale, false)
  assert.equal(f.text, null)
}

{
  const hung = sessionFreshness({ lastOkAtMs: NOW - 30000, nowMs: NOW, recording: true, lastError: '' })
  assert.equal(hung.stale, true)
  assert.equal(hung.ageS, 30)
  assert.doesNotMatch(hung.text, /\(\)/, 'an empty reason renders no empty parentheses')
}

// A clock that jumps backwards is not evidence of freshness.
{
  const f = sessionFreshness({ lastOkAtMs: NOW + 10000, nowMs: NOW, recording: true })
  assert.equal(f.ageS, 0)
  assert.equal(f.stale, false)
}

// Boundary: exactly at the threshold is not yet stale, one ms past it is.
assert.equal(sessionFreshness({ lastOkAtMs: NOW - STALE_AFTER_MS, nowMs: NOW }).stale, true)
assert.equal(sessionFreshness({ lastOkAtMs: NOW - (STALE_AFTER_MS - 1), nowMs: NOW }).stale, false)

// A doubtful number is never rendered bare.
assert.equal(staleSuffix({ stale: true, ageS: 7, text: 'x', tone: 'bad' }), ' · 7s old')
assert.equal(staleSuffix({ stale: false, ageS: 0, text: null, tone: 'warn' }), '')

console.log('sessionFreshness: all assertions passed')
