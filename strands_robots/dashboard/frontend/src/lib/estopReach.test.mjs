import assert from 'node:assert/strict'
import { estopNothingTargeted, signedRailClaim } from '/tmp/estopReach.mjs'

const forbidden = /No live peers were on the mesh|nothing (?:is|was) running|all clear|nothing to stop/i

// --- the defect: an empty target list read as "the room is safe" ---------------
{
  const v = estopNothingTargeted({})
  assert.match(v.headline, /nothing was stopped/, 'lead with what did NOT happen')
  assert.match(v.detail, /statement about this fleet view, not about the room/)
  assert.match(v.detail, /can still be moving/)
  assert.match(v.detail, /cut power at the supply/)
  assert.equal(v.cutPower, true)
  assert.doesNotMatch(`${v.headline} ${v.detail}`, forbidden, 'never reassure from an empty list')
}

// --- skipped-because-unreachable is not "handled" ------------------------------
{
  const v = estopNothingTargeted({ staleSkipped: ['so101-arm-1', 'so101-arm-2'] })
  assert.match(v.headline, /every peer was unreachable/)
  assert.match(v.detail, /so101-arm-1, so101-arm-2/, 'name them; a count is not actionable')
  assert.match(v.detail, /they were.*skipped rather than stopped/)
  assert.match(v.detail, /losing telemetry is not stopping/, 'the exact confusion to prevent')
  assert.doesNotMatch(v.detail, forbidden)

  const one = estopNothingTargeted({ staleSkipped: ['arm'] })
  assert.match(one.detail, /it was.*skipped/, 'singular reads as a sentence, not a template')
}

// --- an empty/garbage list is the same as no list ------------------------------
{
  assert.match(estopNothingTargeted({ staleSkipped: [] }).headline, /no live peer to target/)
  assert.match(estopNothingTargeted({ staleSkipped: ['', null] }).headline, /no live peer to target/)
}

// --- every branch demands power, because that is the one rail the mesh cannot break
for (const v of [estopNothingTargeted({}), estopNothingTargeted({ staleSkipped: ['x'] })]) {
  assert.match(v.detail, /cut power/)
  assert.equal(v.cutPower, true)
}

// =============================================================================
// signedRailClaim: the ISSUER's latch may not be rendered as a claim about the fleet.
// `lockout_engaged` is set unconditionally by Mesh.emergency_stop, so the sheet's
// `peers refuse all commands until resumed` read the same for a stop that reached
// everybody and one that reached nobody.
// =============================================================================

const FLEET_CLAIM = /peers refuse all commands until resumed/

// --- nothing engaged, nothing to claim ----------------------------------------
{
  assert.equal(signedRailClaim({}), null, 'no lockout means no sentence')
  assert.equal(signedRailClaim({ lockoutEngaged: false }), null)
  assert.equal(signedRailClaim({ lockoutEngaged: undefined, responsesReceived: 3 }), null)
}

// --- THE DEFECT: nobody acknowledged must not read like a latched fleet --------
{
  const v = signedRailClaim({ lockoutEngaged: true, issuer: 'dash', responsesReceived: 0 })
  assert.match(v.headline, /NO peer acknowledged/, 'the absence is the headline')
  assert.match(v.detail, /never answered is not a peer that stopped/)
  assert.match(v.detail, /cut power at the supply/)
  assert.equal(v.cutPower, true)
  assert.doesNotMatch(`${v.headline} ${v.detail}`, FLEET_CLAIM,
    'an unacknowledged stop may not claim the peers are refusing commands')
}

// --- a peer that answered "I did not stop" outranks everything else ------------
{
  const v = signedRailClaim({
    lockoutEngaged: true, issuer: 'dash', responsesReceived: 2, peersNotStopped: ['arm-2'],
  })
  assert.match(v.headline, /1 peer reported NOT stopping/, 'singular reads as a sentence')
  assert.match(v.detail, /arm-2/, 'name it; a count is not actionable')
  assert.match(v.detail, /may still be executing/)
  assert.match(v.detail, /stops the NEXT command; it does not halt motion already underway/,
    'the exact thing an operator misreads about a lockout')
  assert.equal(v.cutPower, true)
  assert.doesNotMatch(`${v.headline} ${v.detail}`, FLEET_CLAIM)

  const many = signedRailClaim({
    lockoutEngaged: true, responsesReceived: 3, peersNotStopped: ['arm-2', 'sim-1'],
  })
  assert.match(many.headline, /2 peers reported NOT stopping/)
  assert.match(many.detail, /arm-2, sim-1/)
  assert.match(many.detail, /they did not stop/, 'plural reads as a sentence')
}

// --- a failure to stop outranks the acknowledgement count, even at zero acks ---
{
  const v = signedRailClaim({ lockoutEngaged: true, responsesReceived: 0, peersNotStopped: ['arm-9'] })
  assert.match(v.headline, /NOT stopping/, 'the robot still moving is the worse fact')
  assert.equal(v.cutPower, true)
}

// --- absent accounting is "cannot tell you", not zero and not everyone ---------
{
  const v = signedRailClaim({ lockoutEngaged: true, issuer: 'dash' })
  assert.match(v.headline, /acknowledgements unknown/)
  assert.doesNotMatch(v.headline, /NO peer acknowledged/, 'absent is not zero')
  assert.match(v.detail, /does not report which peers replied/)
  assert.equal(v.cutPower, false, 'unknown is not evidence of motion')
}

// --- the good case may still not overclaim -------------------------------------
{
  const v = signedRailClaim({ lockoutEngaged: true, issuer: 'dash', responsesReceived: 2, peersNotStopped: [] })
  assert.match(v.headline, /2 peers acknowledged, none reported a failure to stop/)
  assert.match(v.detail, /never replied is not counted here/,
    'a silent peer is not covered by the count, and the sentence must say so')
  assert.equal(v.cutPower, false)

  const one = signedRailClaim({ lockoutEngaged: true, responsesReceived: 1 })
  assert.match(one.headline, /1 peer acknowledged/, 'singular reads as a sentence')
}

// --- the issuer is named when known, and never faked --------------------------
{
  assert.match(signedRailClaim({ lockoutEngaged: true, issuer: 'dash', responsesReceived: 1 }).headline,
    /signed by dash/)
  for (const issuer of [undefined, null, '', '   ']) {
    const v = signedRailClaim({ lockoutEngaged: true, issuer, responsesReceived: 1 })
    assert.doesNotMatch(v.headline, /signed by/, `blank issuer ${JSON.stringify(issuer)} must not be rendered`)
    assert.match(v.headline, /fleet LOCKOUT engaged/, 'the latch is still reported')
  }
}

// --- every branch says the lockout IS engaged, because the resume box needs it -
for (const opts of [
  { lockoutEngaged: true, responsesReceived: 0 },
  { lockoutEngaged: true, responsesReceived: 2 },
  { lockoutEngaged: true, responsesReceived: 2, peersNotStopped: ['a'] },
  { lockoutEngaged: true },
]) {
  const v = signedRailClaim(opts)
  assert.match(v.headline, /LOCKOUT engaged/,
    'the resume control is gated on the latch; hiding it would strand a real lockout')
}

// --- garbage peer names are coerced, not crashed on -------------------------
{
  const v = signedRailClaim({ lockoutEngaged: true, responsesReceived: 1, peersNotStopped: [null, '', 'arm-3'] })
  assert.match(v.headline, /1 peer reported NOT stopping/, 'blank entries are not peers')
  assert.match(v.detail, /arm-3/)
}

console.log('estopReach: all assertions passed')
