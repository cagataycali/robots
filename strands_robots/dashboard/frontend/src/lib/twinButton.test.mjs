import assert from 'node:assert/strict'
import { twinButtonCopy } from '/tmp/twinButton.mjs'

const off = twinButtonCopy({ peerId: 'so101-arm-1', twinLive: false })
const on = twinButtonCopy({ peerId: 'so101-arm-1', twinLive: true })
const busy = twinButtonCopy({ peerId: 'so101-arm-1', twinLive: false, busy: true })

// --- the label is READABLE: no lone ideograph, no icon-only button ----------
for (const c of [off, on]) {
  assert.doesNotMatch(c.label, /\u2ffb/, 'the ideograph came back')
  assert.match(c.label, /[a-z]/, 'a label with no letters is not a label')
  assert.ok(c.label.length <= 8, `too long for a card head: ${c.label}`)
  // the accessible name must be a full sentence-ish phrase, not the label again
  assert.ok(c.aria.length > c.label.length + 8)
  assert.match(c.aria, /so101-arm-1/, 'the accessible name must say WHICH robot')
  assert.match(c.title, /so101-arm-1-twin/, 'the title must name the peer it creates or kills')
}

// --- state is legible from the button itself, not only from another card ----
assert.equal(off.cls, '')
assert.equal(on.cls, 'on')
assert.match(off.label, /\+/)
assert.match(on.label, /on/)
assert.match(off.aria, /^start /)
assert.match(on.aria, /^stop /)

// --- the safety claim an operator next to powered arms needs ---------------
assert.match(off.title, /real arm is not touched/)
assert.match(on.title, /real arm is not affected/)
assert.match(off.title, /simulated/)
assert.match(on.title, /mirrored/)

// --- busy says WAITING, and does not lie about which way it is going -------
assert.match(busy.aria, /working/)
assert.match(busy.title, /takes a moment/)
assert.equal(busy.cls, '', 'busy while off must not look like a live twin')
assert.equal(twinButtonCopy({ peerId: 'x', twinLive: true, busy: true }).cls, 'on',
  'busy while on must keep the live look until it is actually gone')

// --- the peer id is interpolated, never assumed ----------------------------
const other = twinButtonCopy({ peerId: 'weird__child', twinLive: false })
assert.match(other.title, /weird__child-twin/)
assert.match(other.aria, /weird__child/)
console.log('twinButtonCopy: all assertions passed')
