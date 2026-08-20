import assert from 'node:assert/strict'
import { estopNothingTargeted } from '/tmp/estopReach.mjs'

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

console.log('estopReach: all assertions passed')
