// The motion confirm renders the interrupt's own words and answers with the exact frame (lib/interruptConfirm.ts).
// Run: npx esbuild src/lib/interruptConfirm.ts --bundle --format=esm --outfile=/tmp/interruptConfirm.mjs \
//        && node src/lib/interruptConfirm.test.mjs
import assert from 'node:assert/strict'

const {
  parseInterruptEvent, parseStatusInterrupt, confirmQuestion, confirmDetail,
  interruptResponseBody, answerNotice,
} = await import('/tmp/interruptConfirm.mjs')

const EV = {
  type: 'interrupt',
  id: 'int-1',
  name: 'physical_motion',
  reason: { tool: 'fleet', action: 'task', target: 'so101-arm-1', instruction: 'wave', duration: 5, why_physical: 'it reports real hardware (so_follower)' },
}

// A well-formed event becomes a confirm carrying the reason's own words.
{
  const c = parseInterruptEvent(EV)
  assert.equal(c.id, 'int-1')
  assert.equal(c.target, 'so101-arm-1')
  assert.equal(c.instruction, 'wave')
  assert.equal(c.duration, 5)
  const q = confirmQuestion(c)
  assert.match(q, /so101-arm-1/)
  assert.match(q, /"wave"/)
  assert.match(q, /5s/)
  assert.match(q, /real motion/)
  assert.match(confirmDetail(c), /real hardware/)
}

// Malformed events are null, never a half-rendered dialog.
assert.equal(parseInterruptEvent(null), null)
assert.equal(parseInterruptEvent({ type: 'token', data: 'x' }), null)
assert.equal(parseInterruptEvent({ type: 'interrupt', id: '' }), null)
assert.equal(parseInterruptEvent({ type: 'interrupt', id: 42 }), null)

// A reason with holes still asks an answerable question instead of rendering "undefined".
{
  const c = parseInterruptEvent({ type: 'interrupt', id: 'int-2', reason: {} })
  assert.equal(c.target, 'a robot')
  assert.match(confirmQuestion(c), /start a task/)
  assert.equal(confirmQuestion(c).includes('undefined'), false)
  assert.equal(confirmDetail(c), '')
}

// Bad durations never render.
for (const d of [-1, 0, NaN, Infinity, 'five']) {
  const c = parseInterruptEvent({ type: 'interrupt', id: 'x', reason: { duration: d } })
  assert.equal(c.duration, null, String(d))
}

// Long durations read as minutes.
{
  const c = parseInterruptEvent({ type: 'interrupt', id: 'x', reason: { target: 'arm', duration: 90 } })
  assert.match(confirmQuestion(c), /1m 30s/)
}

// The reload path (agent_status.interrupt) parses to the same shape.
{
  const c = parseStatusInterrupt({ id: 'int-1', name: 'physical_motion', reason: EV.reason })
  assert.equal(c.id, 'int-1')
  assert.equal(c.target, 'so101-arm-1')
  assert.equal(parseStatusInterrupt(null), null)
  assert.equal(parseStatusInterrupt('junk'), null)
}

// The answer frame is exactly what server.py's parse_chat_frame accepts.
assert.deepEqual(interruptResponseBody('int-1', true), { type: 'interrupt_response', id: 'int-1', response: { approve: true } })
assert.deepEqual(interruptResponseBody('int-1', false), { type: 'interrupt_response', id: 'int-1', response: { approve: false } })

// The transcript notice states the decision and its scope.
{
  const c = parseInterruptEvent(EV)
  assert.match(answerNotice(c, true), /approved/)
  assert.match(answerNotice(c, true), /this once/)
  assert.match(answerNotice(c, false), /nothing was sent/)
}

console.log('interruptConfirm: all assertions passed')
