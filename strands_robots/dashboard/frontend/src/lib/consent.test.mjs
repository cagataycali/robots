// node assertions over the bundled module (esbuild first — see the sibling
// calibrateCommand.test.mjs for the same pattern).
//
// This module decides whether a SECURITY refusal can be answered from the UI, and how loudly. It was
// untested, and its approvability rule was a hardcoded list of one kind while the backend
// (dashboard/consent.py KINDS) had grown to four.
import assert from 'node:assert/strict'
import { findConsent, canApprove, blockedReason, severity, afterApproval } from '/tmp/consent.mjs'

const need = (over = {}) => ({
  kind: 'trust_remote_code', scope: 'trust_remote_code',
  title: 'Run model code from HuggingFace (lerobot_local)?', risk: 'code from the repo executes here',
  env_var: 'STRANDS_TRUST_REMOTE_CODE', grants: ['run repository code for every policy load from now on'],
  ...over,
})

// ── finding the need, wherever the API buried it ──
assert.equal(findConsent(null), null)
assert.equal(findConsent({}), null)
assert.deepEqual(findConsent({ needs_consent: need() }).kind, 'trust_remote_code')
assert.ok(findConsent({ result: { needs_consent: need() } }), 'nested under result')
assert.ok(findConsent({ detail: { result: { needs_consent: need() } } }), 'a 422 body, two deep')
assert.ok(findConsent({ error: { body: { needs_consent: need() } } }))
// a HALF need is not a need: the sheet's whole content comes from these fields
assert.equal(findConsent({ needs_consent: { kind: 'trust_remote_code' } }), null)
assert.equal(findConsent({ needs_consent: { ...need(), risk: undefined } }), null)
// depth is bounded, and a CYCLE must not hang the UI
const cyc = { result: {} }; cyc.result.result = cyc
assert.equal(findConsent(cyc), null)
assert.equal(findConsent({ a: { b: { c: { needs_consent: need() } } } }), null, 'only the known carriers')

// ── approvability: the payload decides, not a hardcoded list ──
assert.equal(canApprove(need()), true)
// THE REGRESSION THAT MATTERED ON THIS FLEET: teleop's degree envelope (Q27) is a complete, grantable
// need, and the old rule disabled its button and blamed an unreadable model name.
assert.equal(canApprove(need({
  kind: 'teleop_degree_units', scope: 'teleop_degree_units', env_var: 'STRANDS_MESH_INPUT_VALUE_ABS',
  grants: ['STRANDS_MESH_INPUT_VALUE_ABS=400', 'STRANDS_MESH_INPUT_SLEW_ABS=800'],
})), true, 'the degree envelope can be granted from the UI')
assert.equal(canApprove(need({ kind: 'agent_physical_motion', env_var: 'STRANDS_AGENT_PHYSICAL_MOTION' })), true)
// a guard invented tomorrow works on arrival, as long as it says what it would set
assert.equal(canApprove(need({ kind: 'some_future_guard', grants: ['does a thing'] })), true)
// …and one that names NOTHING cannot be granted, because approving would grant nothing
assert.equal(canApprove(need({ kind: 'some_future_guard', env_var: undefined, grants: [] })), false)
// hf_repo_allow is the one kind where the grant IS the subject
assert.equal(canApprove(need({ kind: 'hf_repo_allow', subject: 'HashtagRobotics/smolvla' })), true)
assert.equal(canApprove(need({ kind: 'hf_repo_allow', subject: null })), false, 'nothing to allow')
assert.equal(canApprove(need({ kind: 'hf_repo_allow', subject: undefined, env_var: 'X' })), false,
  'an env var does not make a nameless allowlist entry approvable')

// ── the blocked sentence must be TRUE for the kind it explains ──
assert.match(blockedReason(need({ kind: 'hf_repo_allow', subject: null })), /model path/)
const other = blockedReason(need({ kind: 'some_future_guard', env_var: undefined, grants: [] }))
assert.doesNotMatch(other, /model path/, 'a guard with no model path is never explained by one')
assert.match(other, /newer guard|grant it in the environment/)

// ── severity: open-ended capability vs bounded value ──
assert.equal(severity(need()), 'danger', 'arbitrary code execution')
assert.equal(severity(need({ kind: 'agent_physical_motion' })), 'danger',
  'unattended motion on real robots is not level with adding one repository')
assert.equal(severity(need({ kind: 'hf_repo_allow' })), 'warn')
assert.equal(severity(need({ kind: 'teleop_degree_units' })), 'warn', 'an envelope that stays an envelope')
assert.equal(severity(need({ kind: 'brand_new' })), 'danger', 'an unrecognised permission is not routine')

// ── what to say after a grant, and whether a retry can work ──
assert.deepEqual(afterApproval({ granted: false, note: 'refused' }, 'spawn'), { retryNow: false, note: 'refused' })
assert.equal(afterApproval({ granted: false }, 'spawn').retryNow, false)
assert.match(afterApproval({ granted: false }, 'spawn').note, /retrying would fail the same way/)
assert.equal(afterApproval({ granted: true }, 'spawn').retryNow, true, 'the next spawn inherits the grant')
// A RUNNING peer kept the environment it was started with, so an immediate retry would fail the same
// way — the honest answer is respawn first, and it must not be quietly retried.
const peer = afterApproval({ granted: true }, 'peer')
assert.equal(peer.retryNow, false)
assert.match(peer.note, /already running with the old permissions/)
assert.match(peer.note, /Respawn/)
// an already-granted permission counts as granted (a second operator got there first)
assert.equal(afterApproval({ granted: false, already_granted: true }, 'spawn').retryNow, true)
// the server's own note always wins over the local wording
assert.equal(afterApproval({ granted: true, note: 'server says' }, 'peer').note, 'server says')

console.log('consent: all assertions passed')
