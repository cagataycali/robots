import assert from 'node:assert/strict'
import { connBadge } from '/tmp/connBadge.mjs'

const STATES = ['open', 'connecting', 'closed', 'unauthorized']

// --- THE #3 INVARIANT: "LIVE" never stands alone while something is down ----
const meshDown = connBadge('open', { meshDown: true })
assert.notEqual(meshDown.label, 'LIVE')
assert.match(meshDown.label, /page only/)
assert.equal(meshDown.tone, 'warn')
assert.match(meshDown.title, /mesh session is closed/)
assert.match(meshDown.aria, /down/)
// ...but it is still not called OFFLINE: the socket really is open, and a badge
// that lies in the pessimistic direction is just as untrustworthy.
assert.doesNotMatch(meshDown.label, /OFFLINE/)
assert.match(meshDown.label, /LIVE/)

const healthy = connBadge('open', { meshDown: false })
assert.equal(healthy.label, 'LIVE')
assert.equal(healthy.tone, '')
assert.deepEqual(connBadge('open'), healthy, 'no opts means no claim about the mesh')

// --- every badge names its subject, so it cannot be read as a fleet claim ---
for (const s of STATES) {
  for (const meshDown of [false, true]) {
    const b = connBadge(s, { meshDown })
    assert.ok(b.label.length > 0, s)
    assert.ok(/dashboard link/i.test(b.aria), `${s}: aria does not name the subject: ${b.aria}`)
    assert.ok(b.title.length > 20, `${s}: title too thin`)
    assert.ok(['', 'warn', 'bad'].includes(b.tone), `${s}: bad tone ${b.tone}`)
  }
}

// the good state explains what it does NOT cover — the actual #3 complaint
for (const s of ['open', 'connecting', 'closed']) {
  assert.match(connBadge(s).title, /nothing about the robots or the cameras/,
    `${s}: badge still implies the cameras and robots are covered`)
}

// --- the failing states are unmistakable -----------------------------------
assert.equal(connBadge('closed').tone, 'bad')
assert.equal(connBadge('unauthorized').tone, 'bad')
assert.match(connBadge('closed').title, /not updating|nothing on this page is updating/i)
assert.match(connBadge('unauthorized').title, /token/i)
assert.match(connBadge('unauthorized').aria, /not authorised|rejected/i)
assert.equal(connBadge('connecting').tone, 'warn')

// a mesh outage must never DOWNGRADE a worse verdict into something softer
for (const s of ['closed', 'unauthorized']) {
  assert.equal(connBadge(s, { meshDown: true }).tone, 'bad', s)
  assert.deepEqual(connBadge(s, { meshDown: true }), connBadge(s), `${s}: mesh flag changed a hard failure`)
}

// --- an unknown state is never dressed as LIVE -----------------------------
const weird = connBadge('teleporting')
assert.notEqual(weird.label, 'LIVE')
assert.equal(weird.label, 'TELEPORTING')
assert.equal(weird.tone, 'warn')

console.log('connBadge: all assertions passed')
