import assert from 'node:assert/strict'
import { policyLabel, policyGroup, groupPolicies, isKnownPolicy, POLICY_GROUPS } from '/tmp/policyLabels.mjs'

// The 14 providers this fleet's backend actually serves today (GET /api/policies).
const LIVE = ['cosmos3','curobo','groot','kimodo','lerobot_async','lerobot_local','mock',
              'motionbricks','moveit2','protomotions','remote','vera','wbc','wbc_gait']

// --- every live provider has a human name ----------------------------------
for (const n of LIVE) {
  assert.ok(isKnownPolicy(n), `${n}: still shown as a raw identifier`)
  const l = policyLabel(n)
  assert.notEqual(l, n, `${n}: label is the id`)
  assert.ok(l.length > 4 && /[A-Z]/.test(l), `${n}: ${l} does not read as a name`)
  assert.notEqual(policyGroup(n), 'Other', `${n}: ungrouped`)
}

// mock is the one label that must carry a safety promise: it sits next to live
// telemetry, and "mock" alone reads as "this whole screen is fake".
assert.match(policyLabel('mock'), /safe/i)
assert.match(policyLabel('mock'), /no model/i)
assert.equal(policyGroup('mock'), 'Safe test')
assert.equal(POLICY_GROUPS[0], 'Safe test', 'the safe option is offered first')

// the two LeRobot entries must be distinguishable — same family, different rail
assert.notEqual(policyLabel('lerobot_local'), policyLabel('lerobot_async'))
assert.match(policyLabel('lerobot_local'), /local/i)
assert.match(policyLabel('lerobot_async'), /remote|server/i)

// --- a name is NEVER invented ----------------------------------------------
for (const unknown of ['brand_new_policy', 'acme_vla', '', 'GR00T-N2']) {
  assert.equal(policyLabel(unknown), unknown, 'unknown ids must render verbatim')
  assert.equal(isKnownPolicy(unknown), false)
  assert.equal(policyGroup(unknown), 'Other')
}

// --- grouping may never lose a policy --------------------------------------
const withNew = [...LIVE, 'brand_new_policy']
const grouped = groupPolicies(withNew.map(name => ({ name })), p => p.name)
const flat = grouped.flatMap(g => g.items.map(i => i.name))
assert.deepEqual([...flat].sort(), [...withNew].sort(), 'a provider vanished from the dropdown')
assert.equal(flat.length, new Set(flat).size, 'a provider was listed twice')
// fixed group order, empty groups dropped, and the unknown one is visible last
const order = grouped.map(g => g.group)
assert.deepEqual(order, POLICY_GROUPS.filter(g => order.includes(g)))
assert.equal(order[0], 'Safe test')
assert.equal(order[order.length - 1], 'Other')
// backend order preserved inside a group
const learned = grouped.find(g => /Learned/.test(g.group)).items.map(i => i.name)
assert.deepEqual(learned, LIVE.filter(n => /Learned/.test(policyGroup(n))))
// empty input is not a crash
assert.deepEqual(groupPolicies([], (x) => String(x)), [])

// --- labels are distinct: two identical lines in a dropdown is a coin toss --
const labels = LIVE.map(policyLabel)
assert.equal(labels.length, new Set(labels).size, 'duplicate labels')

console.log('policyLabels: all assertions passed')
