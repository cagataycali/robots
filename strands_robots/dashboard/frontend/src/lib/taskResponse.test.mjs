// Run: node scripts/run-lib-tests.mjs taskResponse lib/taskOutcome covers the THROWN cases
// (whether the arm may be moving after a rejected fetch). This is the other half: a response
// that ARRIVED, and what it actually says.
import assert from 'node:assert/strict'

const { interpretRun, interpretStop, errorInResult } = await import('/tmp/taskResponse.mjs')

// ── 1. the error a peer's reply carries, at either nesting depth ──
assert.equal(errorInResult({ error: 'boom' }), 'boom', 'the bridge result itself')
assert.equal(errorInResult({ result: { error: 'boom' } }), 'boom', 'a peer that wraps its tool result')
assert.equal(errorInResult({ error: { code: 7 } }), '[object Object]',
             'a non-string error still counts as an error (the readable payload is in `detail`) — dropping ' +
             'it would let a structured refusal read as a successful start')
assert.equal(errorInResult({ ok: true, started: true }), undefined, 'a clean payload has no error')
assert.equal(errorInResult(undefined), undefined, 'no payload at all does not throw')
assert.equal(errorInResult({ error: '' }), undefined, 'an EMPTY error is not an error — it would print as blank')
assert.equal(errorInResult({ error: null }), undefined, 'nor is a null one')

// ── 2. a clean start ──
const started = interpretRun({ ok: true, result: { ok: true } })
assert.equal(started.phase, 'running')
assert.deepEqual(started.outcome, { ok: true, text: 'running' })
assert.equal(interpretRun({ ok: true, result: {}, routed_to: 'so101-follower' }).outcome.text, 'running via so101-follower',
             'routing is worth saying: the operator asked one peer and another carried it out')
assert.equal(interpretRun({ ok: true, result: {}, mirrored_to_twin: true }).outcome.text, 'running + twin')

// ── 3.
const jammed = interpretRun({ ok: true, result: { error: 'gripper jammed' } })
assert.equal(jammed.outcome.ok, false, 'Q88: the PEER said no, so the answer is no — whatever the envelope says')
assert.equal(jammed.outcome.text, 'gripper jammed', 'and the peer\'s own words are the news, not "refused"')
assert.equal(jammed.phase, 'failed', 'the phase must not go to running, or stop/▶ offer the wrong action')
assert.equal(interpretRun({ ok: true, result: { result: { error: 'no policy loaded' } } }).outcome.text,
             'no policy loaded', 'the nested shape too — that ?? chain existed because both are real')

// A refusal the envelope DOES report keeps working, with the payload as detail.
const refused = interpretRun({ ok: false, result: { detail: 'peer exposes no run_task' } })
assert.equal(refused.outcome.text, 'refused', 'no error message in the payload = the generic word')
assert.equal(refused.outcome.detail, '{"detail":"peer exposes no run_task"}', 'the payload is shown verbatim')
assert.equal(refused.phase, 'failed')

const noResult = interpretRun({ ok: false })
assert.equal(noResult.outcome.ok, false)
assert.equal(noResult.outcome.detail, 'no result payload', 'says so in words rather than throwing')
assert.notEqual(noResult.outcome.ambiguous, true, 'a flat refusal must NOT be dressed as "may be moving"')
assert.equal(interpretRun(undefined).outcome.ok, false, 'an empty body is a refusal, not a start')
assert.equal(interpretRun({}).phase, 'failed', 'and an envelope with no ok at all is not a start either')

// ── 5. stop: never a bare "stopped" on silence ──
const stopped = interpretStop({ peer_id: 'a', state: 'stopped' })
assert.deepEqual(stopped.outcome, { ok: true, text: 'stopped' })
assert.equal(stopped.phase, 'idle', 'only a real stop returns the card to idle')

const silent = interpretStop({ peer_id: 'a', state: 'no_answer' })
assert.equal(silent.outcome.ok, false, 'a stop that got no answer is not a stop')
assert.match(silent.outcome.text, /may still be moving/, 'and it says the dangerous part out loud')
assert.equal(silent.outcome.ambiguous, true,
             'ambiguous is what makes the UI render it louder — the two demand different behaviour from a ' +
             'human standing next to the hardware')
assert.equal(silent.phase, 'failed')

assert.equal(interpretStop({ peer_id: 'a', state: 'not_stopped', detail: 'no stop_task' }).outcome.text,
             'not stopped: no stop_task', 'a string detail is shown as-is')
assert.equal(interpretStop({ peer_id: 'a', state: 'not_stopped', detail: { why: 'busy' } }).outcome.text,
             'not stopped: {"why":"busy"}', 'a structured detail is rendered, not printed as [object Object]')
assert.equal(interpretStop({ peer_id: 'a', state: 'not_stopped' }).outcome.text, 'not stopped: {}',
             'no detail at all still says clearly that it did not stop')

// An unrecognised state (an older or newer server) must not read as success.
for (const state of [undefined, 'queued', '']) {
  const v = interpretStop({ peer_id: 'a', state })
  assert.equal(v.outcome.ok, false, `state ${JSON.stringify(state)} is not a stop`)
  assert.equal(v.phase, 'failed', 'and it does not return the card to idle')
}
assert.equal(interpretStop(undefined).outcome.ok, false, 'an empty body is not a stop')

console.log('taskResponse.test.mjs: all assertions passed')
