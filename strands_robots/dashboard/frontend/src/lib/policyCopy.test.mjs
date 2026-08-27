import assert from 'node:assert/strict'
import { fieldCopy, requirementSummary, missingSummary, localOnlySummary } from '/tmp/policyCopy.mjs'

// --- rule 1: curated keys read as English -----------------------------------
const ck = fieldCopy('pretrained_name_or_path')
assert.equal(ck.known, true)
assert.equal(ck.label, 'checkpoint')
assert.match(ck.hint, /Hugging Face/, 'the hint must say what the operator needs to HAVE')

assert.equal(fieldCopy('policy_type').label, 'policy family')
assert.match(fieldCopy('policy_type').hint, /act|smolvla/, 'name real values, not "a string"')
assert.match(fieldCopy('policy_port').hint, /already be running/, 'a port implies a server exists')

// aliases and their registry originals agree — the UI must not call the same
// thing two names depending on which provider is selected
assert.equal(fieldCopy('port').label, fieldCopy('policy_port').label)
assert.equal(fieldCopy('host').label, fieldCopy('policy_host').label)
assert.equal(fieldCopy('repo_id').label, 'checkpoint')

// --- rule 2: an unknown identifier is printed VERBATIM, never guessed -------
const unknown = fieldCopy('norm_tag')
assert.equal(unknown.known, false)
assert.equal(unknown.label, 'norm_tag', 'no snake_case-to-prose invention')
assert.equal(unknown.hint, undefined, 'no invented explanation either')
assert.equal(fieldCopy('').label, '')

// --- the <option> one-liner --------------------------------------------------
assert.equal(
  requirementSummary(['policy_type', 'pretrained_name_or_path']),
  'policy family + checkpoint',
)
assert.equal(requirementSummary([]), '', 'a provider that needs nothing says nothing')
assert.equal(requirementSummary(['port']), 'server port')
// duplicate labels collapse: "a checkpoint + a checkpoint" reads as two things
assert.equal(requirementSummary(['checkpoint', 'repo_id']), 'checkpoint')
// an unknown key still appears, as itself
assert.equal(requirementSummary(['norm_tag', 'port']), 'norm_tag + server port')

// --- the blocking-fields sentence keeps the identifier searchable -----------
const miss = missingSummary(['pretrained_name_or_path', 'policy_type'])
assert.equal(miss, 'checkpoint (pretrained_name_or_path), policy family (policy_type)')
assert.match(miss, /pretrained_name_or_path/, 'the identifier is what they paste into a script')
// unknown key must not render as "foo (foo)"
assert.equal(missingSummary(['norm_tag']), 'norm_tag')

// --- local-only kwargs list -------------------------------------------------
assert.equal(
  localOnlySummary(['device', 'norm_tag']),
  'compute device (device), norm_tag',
)

// every curated entry is non-empty and is NOT just the identifier back again
for (const key of ['pretrained_name_or_path', 'policy_type', 'port', 'host', 'device',
  'data_config', 'action_horizon', 'target_pose', 'trust_remote_code']) {
  const c = fieldCopy(key)
  assert.equal(c.known, true, `${key} should be curated`)
  assert.ok(c.label.length > 0)
  assert.notEqual(c.label, key, `${key}: a "label" identical to the identifier is not a label`)
}

console.log('policyCopy: all assertions passed')
