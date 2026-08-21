import assert from 'node:assert/strict'
import { busRecoveryBadge, BUS_RECOVERY_WARN_AT } from '/tmp/busRecoveries.mjs'

// --- a healthy arm earns no ornament ----------------------------------------
assert.equal(busRecoveryBadge(0), null, 'zero must render NOTHING, not "bus healed ×0"')
assert.equal(busRecoveryBadge(undefined), null, 'an older peer that publishes no count is not a fault')
assert.equal(busRecoveryBadge(null), null)
assert.equal(busRecoveryBadge('3'), null, 'a non-number is not evidence - do not invent a count')
assert.equal(busRecoveryBadge(-2), null)

// --- one hiccup: say it, and say it needs nothing ---------------------------
{
  const b = busRecoveryBadge(1)
  assert.equal(b.label, 'bus healed once', 'read as English at a glance, not "×1"')
  assert.equal(b.tone, '', 'one stranding is not a warning')
  assert.match(b.title, /joints below are real/, 'reassure: the reading on screen is trustworthy')
  assert.match(b.title, /hiccup and needs nothing/)
  assert.match(b.title, /USB cable|hub|connector/, 'name the physical causes')
}

// --- a pattern: escalate, and say what to DO --------------------------------
{
  const b = busRecoveryBadge(BUS_RECOVERY_WARN_AT)
  assert.equal(b.tone, 'warn', 'at the threshold it stops being bad luck')
  assert.equal(b.label, `bus healed ×${BUS_RECOVERY_WARN_AT}`)
  assert.match(b.title, /swap the cable/, 'an escalation with no action is just anxiety')
  assert.match(b.title, /powered hub|different port/)
  assert.match(b.title, /dataset/, 'connect it to the thing the user is actually trying to do')
}

// --- the count is the message, so it must survive to the label --------------
assert.equal(busRecoveryBadge(23).label, 'bus healed ×23')
assert.equal(busRecoveryBadge(4).tone, '', 'just below the threshold stays quiet')
assert.equal(busRecoveryBadge(2.7).label, 'bus healed ×2', 'floor a fractional count, never round up')

console.log('busRecoveries: ok')
