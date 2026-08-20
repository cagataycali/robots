import assert from 'node:assert/strict'
import { shouldCheckForUpdate, bundleAgeText, SW_UPDATE_INTERVAL_MS } from '/tmp/swUpdate.mjs'

const NOW = 1_787_200_000_000
const base = { nowMs: NOW, online: true, visible: true, reason: 'interval' }

// --- the incident: a page open for 11 hours had never re-checked ---------------
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW - 11 * 3600_000 }), true)
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: null }), true,
  'never checked since load is exactly the state that stranded the phone')

// --- but not a request per tick -----------------------------------------------
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW - 60_000 }), false)
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW - SW_UPDATE_INTERVAL_MS + 1 }), false)
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW - SW_UPDATE_INTERVAL_MS }), true)

// --- foregrounding is a reason to ask, but does not bypass the interval --------
assert.equal(shouldCheckForUpdate({ ...base, reason: 'visible', lastCheckedAt: NOW - 30_000 }), false,
  'app-switching on a phone must not become a request per switch')
assert.equal(shouldCheckForUpdate({ ...base, reason: 'visible', lastCheckedAt: NOW - 3600_000 }), true)

// --- a phone's realities ------------------------------------------------------
assert.equal(shouldCheckForUpdate({ ...base, online: false, lastCheckedAt: null }), false,
  'a check that cannot succeed is battery and noise')
assert.equal(shouldCheckForUpdate({ ...base, visible: false, lastCheckedAt: null }), false,
  'hidden tabs are throttled; asking there just queues work for later')

// --- registration always checks (it is the baseline) --------------------------
assert.equal(shouldCheckForUpdate({ ...base, reason: 'registered', lastCheckedAt: NOW, online: false, visible: false }), true)

// --- a clock that moved backwards cannot wedge this either way ----------------
assert.equal(shouldCheckForUpdate({ ...base, lastCheckedAt: NOW + 3600_000 }), true,
  'a future last-check must not block updates forever')

// --- how old the running bundle is, in human words ---------------------------
assert.equal(bundleAgeText(NOW - 5_000, NOW), 'just now')
assert.equal(bundleAgeText(NOW - 600_000, NOW), '10m ago')
assert.equal(bundleAgeText(NOW - 11 * 3600_000, NOW), '11.0h ago')
assert.equal(bundleAgeText(NOW - 3 * 86400_000, NOW), '3d ago')
assert.equal(bundleAgeText(null, NOW), null, 'unknown stays unknown, never "just now"')
assert.equal(bundleAgeText(NOW + 10_000, NOW), null)

console.log('swUpdate: all assertions passed')
