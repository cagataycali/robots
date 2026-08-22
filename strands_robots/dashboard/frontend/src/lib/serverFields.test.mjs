import assert from 'node:assert/strict'
const { fieldSupport } = await import('/tmp/serverFields.mjs')

// Known and present.
assert.deepEqual(fieldSupport(['steps', 'val_episodes'], 'val_episodes', true), { ok: true, why: '' })

// Known and absent: refused HERE, and the reason is an action.
{
  const s = fieldSupport(['steps', 'method'], 'val_episodes', true)
  assert.equal(s.ok, false)
  assert.match(s.why, /does not accept val_episodes/)
  assert.match(s.why, /Restart the dashboard/, 'the remedy, not just the diagnosis')
}

// No list at all = a server older than the list itself, so older than the field too.
for (const missing of [undefined, null, {}, 'steps,method']) {
  const s = fieldSupport(missing, 'val_episodes', true)
  assert.equal(s.ok, false, `${JSON.stringify(missing)} is not a field list`)
  assert.match(s.why, /older than val_episodes/)
  assert.match(s.why, /Restart the dashboard/)
}

// Not loaded yet, or the fetch failed: stay silent. Disabling a field because OUR request failed
// turns a network hiccup into a missing feature — the server's own refusal is the backstop.
assert.deepEqual(fieldSupport(null, 'val_episodes', false), { ok: true, why: '' })
assert.deepEqual(fieldSupport(['steps'], 'val_episodes', false), { ok: true, why: '' })

console.log('serverFields: all assertions passed')
