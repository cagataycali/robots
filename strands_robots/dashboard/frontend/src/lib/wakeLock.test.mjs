// Run: node scripts/run-lib-tests.mjs wakeLock The screen wake lock is the difference between
// watching a moving arm and staring at a black phone. The decision was three inline conditions
// inside usePwa, reachable only from a real browser.
import assert from 'node:assert/strict'

const { wakeLockAction, wakeLockNote } = await import('/tmp/wakeLock.mjs')

const at = (o) => wakeLockAction({ want: false, held: false, visible: true, supported: true, ...o })

// ── 1. the ordinary arc ──
assert.equal(at({ want: true }), 'request', 'a robot started and we hold nothing: take the lock')
assert.equal(at({ want: true, held: true }), 'none', 'already held — asking again would stack locks')
assert.equal(at({ want: false, held: true }), 'release',
             'the task ended: give the screen back, or the phone burns its battery for the next task')
assert.equal(at({}), 'none', 'nothing wanted, nothing held')

// ── 2.
assert.equal(at({ want: true, held: false, visible: true }), 'request',
             'Q89: back in the foreground, still running, lock gone -> TAKE IT AGAIN')
assert.equal(at({ want: true, held: false, visible: false }), 'none',
             'but not while hidden: the browser refuses a request from a hidden page (NotAllowedError), ' +
             'and a burned request leaves held false with nothing to show for it')
assert.equal(at({ want: true, held: true, visible: false }), 'none',
             'hidden and still nominally held: nothing to do — the release event is what tells us it went')

// A release does NOT need visibility: the task ended while the operator was in another app.
assert.equal(at({ want: false, held: true, visible: false }), 'release',
             'releasing works from a hidden page, and holding on after the task is over is a battery leak')

// ── 3. an unsupported platform is never pretended into working ── navigator.wakeLock is
// absent on Firefox and on iOS Safari before 16.4 — a real share of the phones this cockpit
// gets opened on.
for (const s of [{ want: true }, { want: true, held: true }, { want: false, held: true }]) {
  assert.equal(at({ ...s, supported: false }), 'none', 'no API to call, so no action is invented')
}

// ── 4. what the operator is told — the truth from the API, never our intent ──
const note = (o) => wakeLockNote({ want: false, held: false, visible: true, supported: true, ...o })
assert.equal(note({}), null, 'nothing running: the screen is not the operator\'s problem')
assert.equal(note({ want: true, held: true }), null, 'held: silence is the correct UI — it works')
assert.match(note({ want: true, supported: false }), /cannot keep the screen awake/,
             'an unsupported browser says so, because the operator can act on it (plug in, do not lock)')
assert.match(note({ want: true, held: false }), /not being prevented yet/,
             'wanted but not held (denied, or waiting to return to the foreground) must not read as held')
assert.equal(note({ want: false, held: true, supported: false }), null,
             'no news when nothing is running, even on a platform that cannot help')

// ── 5. the wiring, because the pure table cannot see it ── Everything above is true of a
// function nobody calls.
import { readFileSync } from 'node:fs'
const src = readFileSync(new URL('./usePwa.ts', import.meta.url), 'utf8')
assert.match(src, /addEventListener\('visibilitychange', onVisible\)/,
             'usePwa must listen for visibility changes')
assert.match(src, /const onVisible = \(\) => \{ void applyWakeLock\(\) \}/,
             'Q89: and that listener must re-apply the WAKE LOCK — the update-check listener is a different one')
assert.match(src, /wantAwakeRef\.current = want/,
             'the desired state has to be REMEMBERED, or there is nothing to re-apply')
assert.doesNotMatch(src, /navigator\.wakeLock\.request|anyNav\.wakeLock\.request\('screen'\)[\s\S]{0,40}keepAwake/,
             'the request must stay behind wakeLockAction, not be re-inlined into keepAwake')

console.log('wakeLock.test.mjs: all assertions passed')
