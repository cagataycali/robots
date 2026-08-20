// Assertions for the camera socket's retry policy (lib/cameraRetry.ts).
// Run: npx esbuild src/lib/cameraRetry.ts --bundle --format=esm --outfile=/tmp/cameraRetry.mjs \
//        && node src/lib/cameraRetry.test.mjs
import assert from 'node:assert/strict'

const { planRetry, backoffMs, MAX_RETRY_MS, MIN_USEFUL_OPEN_MS } = await import('/tmp/cameraRetry.mjs')

// THE MEASURED INCIDENT (Q40): a socket that is accepted and then closed with nothing
// sent must be counted as a FAILURE. This is the whole bug: onopen reset the counter,
// so the delay stayed at 1s and a phone on cellular opened 63,906 sockets in 10 hours.
{
  let attempt = 0
  let elapsed = 0
  let opens = 0
  const TEN_HOURS = 10 * 3600 * 1000
  while (elapsed < TEN_HOURS) {
    opens += 1
    // exactly the live behaviour: handshake accepted, open ~50ms, zero frames
    const plan = planRetry({ attempt, frames: 0, openMs: 50, code: 1000 })
    assert.notEqual(plan.delayMs, null)
    attempt = plan.attempt
    elapsed += 50 + plan.delayMs
  }
  assert.ok(opens < 1500, `10h of a dead camera must not be 63,906 sockets — got ${opens}`)
  assert.ok(opens > 100, 'it must still keep trying, so a camera coming back is noticed')
}

// A socket that delivered frames and then dropped IS a blip: recover fast.
{
  const plan = planRetry({ attempt: 7, frames: 240, openMs: 60_000 })
  assert.equal(plan.attempt, 1)
  assert.equal(plan.delayMs, 1000)
  assert.match(plan.reason, /delivered frames/)
}

// Staying open a long while with no frames still counts as working: some cameras are
// slow to publish, and punishing them would leave the tile dark for half a minute.
{
  const plan = planRetry({ attempt: 5, frames: 0, openMs: MIN_USEFUL_OPEN_MS })
  assert.equal(plan.attempt, 1)
  assert.match(plan.reason, /stayed open long enough/)
}

// One millisecond short of that is not evidence.
{
  const plan = planRetry({ attempt: 5, frames: 0, openMs: MIN_USEFUL_OPEN_MS - 1 })
  assert.equal(plan.attempt, 6)
  assert.match(plan.reason, /counted as a failure/)
}

// A refusal is an answer, not a race to lose.
{
  const plan = planRetry({ attempt: 3, frames: 0, openMs: 10, code: 1008 })
  assert.equal(plan.delayMs, null)
  assert.match(plan.reason, /refused/)
}

// A socket that never opened at all is still a failure that must accumulate.
{
  const plan = planRetry({ attempt: 2, frames: 0 })
  assert.equal(plan.attempt, 3)
  assert.match(plan.reason, /never opened/)
}

// The curve: exponential, floored at 1s, ceilinged so a camera coming back is still
// picked up within half a minute.
{
  assert.equal(backoffMs(0), 1000)
  assert.equal(backoffMs(1), 1000)
  assert.equal(backoffMs(2), 2000)
  assert.equal(backoffMs(5), 16000)
  assert.equal(backoffMs(99), MAX_RETRY_MS)
}

console.log('cameraRetry: all assertions passed')

// --- Q51: a stream that keeps dying after a few frames is churn, not health ---------
// MEASURED: one cellular phone reopened /ws/camera/so101-arm-1/top 1.55x/s for 11 hours,
// while the server side of that very tile was healthy (42 frames / 4.1 MB in 9s from
// localhost = 0.45 MB/s). Those sockets almost certainly DID carry a frame or two, and
// `frames > 0` reset the delay to 1s — the generous rule is what sustained the storm.
{
  const { CHURN_FLOOR_MS, CHURN_OPENS_PER_MIN } = await import('/tmp/cameraRetry.mjs')
  let attempt = 0, elapsed = 0, opens = 0
  const window_ = []
  const HOUR = 3600 * 1000
  while (elapsed < HOUR) {
    opens += 1
    window_.push(elapsed)
    while (window_.length && elapsed - window_[0] > 60_000) window_.shift()
    // the live shape: opens, delivers 2 frames, link gives up after ~650ms
    const plan = planRetry({ attempt, frames: 2, openMs: 650, code: 1006, recentOpens: window_.length })
    assert.notEqual(plan.delayMs, null)
    attempt = plan.attempt
    elapsed += 650 + plan.delayMs
  }
  assert.ok(opens < 700, `an hour of a link that cannot keep up must not be ~5,600 sockets — got ${opens}`)
  assert.ok(opens > 300, 'and the tile must still refresh every few seconds, not give up on a working camera')
}

// A genuine blip still resets: frames + a long life is a healthy stream, not churn.
{
  const p = planRetry({ attempt: 5, frames: 900, openMs: 200_000, code: 1006, recentOpens: 1 })
  assert.equal(p.attempt, 1)
  assert.equal(p.delayMs, 1000)
}

// Churn NEVER outranks a refusal: 1008 must still stop dead.
{
  const p = planRetry({ attempt: 9, frames: 3, openMs: 300, code: 1008, recentOpens: 40 })
  assert.equal(p.delayMs, null)
}

// The floor applies to the zero-frame storm too, and never below the escalated backoff.
{
  const early = planRetry({ attempt: 1, frames: 0, openMs: 50, code: 1000, recentOpens: 6 })
  assert.ok(early.delayMs >= 5000, 'floor applies once the tile is demonstrably churning')
  const late = planRetry({ attempt: 12, frames: 0, openMs: 50, code: 1000, recentOpens: 30 })
  assert.equal(late.delayMs, 30_000, 'an established backoff must not be lowered to the floor')
}

// Below the churn threshold nothing changes — one reconnect is not a storm.
{
  const p = planRetry({ attempt: 0, frames: 1, openMs: 400, code: 1006, recentOpens: 2 })
  assert.equal(p.attempt, 1)
  assert.equal(p.delayMs, 1000)
}
console.log('cameraRetry: Q51 churn assertions ok')
