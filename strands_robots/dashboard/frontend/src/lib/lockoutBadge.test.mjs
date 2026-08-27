import assert from 'node:assert/strict'

const { lockoutBadge, lockoutBanner } = await import('/tmp/lockoutBadge.mjs')

const NOW = 1_700_000_000_000

{
  const b = lockoutBadge({ state: 'locked', by: 'evac-coordinator', since: NOW / 1000 - 36000,
                           reason: 'an e-stop from evac-coordinator locked the fleet' }, NOW)
  assert.equal(b.label, 'e-stop locked')
  assert.equal(b.tone, 'locked')
  assert.match(b.title, /evac-coordinator/)
  assert.match(b.title, /10h ago/)
  assert.match(b.title, /override code/)
  // it must say what resume does NOT do - an operator reading "resume" fears motion
  assert.match(b.title, /moves nothing/)
}

// THE CRY-WOLF TRAP: every peer is 'unknown' on an ordinary fresh start, because the mesh does
// not advertise lockout state.
{
  const b = lockoutBadge({ state: 'unknown', reason: 'no e-stop or resume seen since this dashboard started' }, NOW)
  assert.equal(b.label, null, 'silence, not a warning, when nothing has happened')
  assert.equal(b.tone, null)
}

// Unknown WITH an event behind it is meaningful doubt and must be shown.
{
  const b = lockoutBadge({ state: 'unknown', since: NOW / 1000 - 120,
                           reason: 'a resume was broadcast, but each peer verifies the override code itself' }, NOW)
  assert.equal(b.label, 'e-stop?')
  assert.equal(b.tone, 'doubt')
  assert.match(b.title, /2m ago/)
  assert.match(b.title, /accepts a command/, 'say how it resolves, so the doubt is actionable')
}

// 'clear' renders NOTHING: a green "safe" badge is one more claim to trust.
assert.equal(lockoutBadge({ state: 'clear', reason: 'a command this peer accepted proves it' }, NOW).label, null)

// A server too old to send the field must not produce a badge either way.
assert.equal(lockoutBadge(undefined, NOW).label, null)
assert.equal(lockoutBadge(null, NOW).label, null)
assert.equal(lockoutBadge({}, NOW).label, null)

// Fresh lockout wording stays readable in seconds and minutes too.
assert.match(lockoutBadge({ state: 'locked', since: NOW / 1000 - 5 }, NOW).title, /5s ago/)
assert.match(lockoutBadge({ state: 'locked', since: NOW / 1000 - 600 }, NOW).title, /10m ago/)

// ---- the fleet banner ----
{
  const banner = lockoutBanner([
    { peer_id: 'so101-arm-1', lockout: { state: 'locked', by: 'evac-coordinator', since: 1 } },
    { peer_id: 'so101-arm-2', lockout: { state: 'locked', by: 'evac-coordinator', since: 1 } },
    { peer_id: 'gateway-x', lockout: { state: 'unknown' } },
  ])
  assert.equal(banner.severity, 'bad')
  assert.match(banner.text, /2 robots/)
  assert.match(banner.text, /so101-arm-1, so101-arm-2/, 'name them: "some robots" sends the operator hunting')
  assert.match(banner.text, /evac-coordinator/)
}
{
  const one = lockoutBanner([{ peer_id: 'so101-arm-2', lockout: { state: 'locked', since: 1 } }])
  assert.match(one.text, /so101-arm-2 is e-stop locked/)
}
// Doubt alone is a warning, not an alarm.
{
  const banner = lockoutBanner([{ peer_id: 'a', lockout: { state: 'unknown', since: 2 } }])
  assert.equal(banner.severity, 'warn')
  assert.match(banner.text, /not proof/)
}
// A healthy fleet says nothing at all.
assert.equal(lockoutBanner([{ peer_id: 'a', lockout: { state: 'clear' } }, { peer_id: 'b' }]), null)
assert.equal(lockoutBanner([]), null)
// A locked peer OUTRANKS doubt: the loudest true thing wins.
assert.equal(lockoutBanner([
  { peer_id: 'a', lockout: { state: 'unknown', since: 2 } },
  { peer_id: 'b', lockout: { state: 'locked', since: 1 } },
]).severity, 'bad')

console.log('lockoutBadge: all assertions passed')
