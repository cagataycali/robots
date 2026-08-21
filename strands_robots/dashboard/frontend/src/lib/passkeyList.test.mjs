import assert from 'node:assert/strict'
import { passkeyRows, revokeRefusal, passkeySummary } from '/tmp/passkeyList.mjs'

const NOW = 1_700_000_000_000

// Two keys: both revocable, both named, times phrased like the rest of the UI.
const rows = passkeyRows([
  { id: 'a', name: "cagatay's iPhone", created: (NOW - 3600_000) / 1000 },
  { id: 'b', name: '', created: null },
], NOW)
assert.equal(rows.length, 2)
assert.equal(rows[0].label, "cagatay's iPhone")
assert.match(rows[0].when, /^added .*ago$/)
assert.equal(rows[1].label, 'passkey')          // unnamed is not blank
assert.equal(rows[1].when, '')                  // no date invented from silence
assert.ok(rows.every(r => r.revocable && r.reason === ''))

// ONE key: the server would refuse, so the button must explain BEFORE the click.
const one = passkeyRows([{ id: 'a', name: 'iPad', created: NOW / 1000 }], NOW)
assert.equal(one[0].revocable, false)
assert.match(one[0].reason, /only key/)
assert.match(one[0].reason, /enroll another/)

// Milliseconds in a seconds field must not render as millennia.
assert.match(passkeyRows([{ id: 'a', created: NOW }, { id: 'b' }], NOW)[0].when, /just now|added/)
assert.equal(passkeyRows([{ id: 'a', created: 0 }, { id: 'b' }], NOW)[0].when, '')
assert.equal(passkeyRows([{ id: 'a', created: 'nonsense' }, { id: 'b' }], NOW)[0].when, '')
// a clock skew that puts enrollment in the future is not a negative age
assert.equal(passkeyRows([{ id: 'a', created: (NOW + 9e6) / 1000 }, { id: 'b' }], NOW)[0].when, 'just now')

// Junk from the wire cannot produce a row with no id (its revoke would DELETE /undefined).
assert.deepEqual(passkeyRows(null), [])
assert.deepEqual(passkeyRows(undefined), [])
assert.deepEqual(passkeyRows([{ name: 'no id' }, null, { id: '' }]), [])

// The empty case is about EXPOSURE, and it says which kind.
assert.match(passkeySummary([], false), /anyone who can reach this page/)
assert.match(passkeySummary([], true), /nobody can sign in/)
assert.equal(passkeySummary(one, true), '1 device can sign in to this dashboard')
assert.equal(passkeySummary(rows, true), '2 devices can sign in to this dashboard')

assert.match(revokeRefusal(409, ''), /last passkey/)
assert.equal(revokeRefusal(409, 'cannot remove the last passkey - enroll another first'),
  'cannot remove the last passkey - enroll another first')
assert.match(revokeRefusal(404, ''), /already gone/)
assert.match(revokeRefusal(401, ''), /session expired/)
assert.match(revokeRefusal(500, ''), /could not remove/)
console.log('passkeyList: all assertions passed')
