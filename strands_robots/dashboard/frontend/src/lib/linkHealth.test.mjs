import assert from 'node:assert/strict'
import { linkHealth, estopPosture } from '/tmp/linkHealth.mjs'

const base = { browserOnline: true, now: 10_000, peerCount: 2, everOpen: true, lastEventAt: 1_000 }

// the ordinary case says nothing at all
const live = linkHealth({ ...base, conn: 'open' })
assert.equal(live.kind, 'live')
assert.equal(live.commandsWork, true)
assert.equal(live.headline, '', 'a healthy link must be silent — a banner that is always up is furniture')

// THE MEASURED HOLE: socket dropped while robots are on screen
const lost = linkHealth({ ...base, conn: 'closed' })
assert.equal(lost.kind, 'lost')
assert.equal(lost.commandsWork, false)
assert.equal(lost.misleading, true, 'peers on screen that cannot be commanded IS the danger')
assert.match(lost.detail, /frozen/)
assert.match(lost.detail, /STOP ALL cannot reach them/)
assert.match(lost.detail, /power switch/, 'name the physical fallback, every time')
assert.match(lost.detail, /9s old/, 'how stale, in seconds, not "a while ago"')

// same break with an EMPTY fleet misleads nobody: no frozen-fleet wording
const lostEmpty = linkHealth({ ...base, conn: 'closed', peerCount: 0 })
assert.equal(lostEmpty.misleading, false)
assert.doesNotMatch(lostEmpty.detail, /frozen|power switch/)

// startup connecting is not a warning; a RECONNECT after an open one is
const boot = linkHealth({ ...base, conn: 'connecting', everOpen: false, lastEventAt: undefined })
assert.equal(boot.kind, 'connecting')
assert.equal(boot.headline, '', 'first connect must not flash a scary banner')
const recon = linkHealth({ ...base, conn: 'connecting' })
assert.equal(recon.kind, 'lost')
assert.match(recon.headline, /Reconnecting/)

// this device's network outranks the server's health
const off = linkHealth({ ...base, conn: 'open', browserOnline: false })
assert.equal(off.kind, 'device-offline')
assert.match(off.detail, /cached snapshot/)
assert.match(off.detail, /power switch/)

// a refused token is a third, different world
const un = linkHealth({ ...base, conn: 'unauthorized' })
assert.equal(un.kind, 'unauthorized')
assert.match(un.detail, /refused/)

// an open-but-mute socket: suspicious only with robots on screen
assert.equal(linkHealth({ ...base, conn: 'open', lastEventAt: 10_000 - 25_000 }).kind, 'stalled')
assert.equal(linkHealth({ ...base, conn: 'open', lastEventAt: 10_000 - 25_000, peerCount: 0 }).kind, 'live',
  'an idle empty dashboard must not cry stalled — that teaches the operator to ignore this banner')
assert.equal(linkHealth({ ...base, conn: 'open', lastEventAt: undefined }).kind, 'live',
  'never having received a frame is not evidence of a stall')
// a stall still lets commands through: the socket is up, only the stream is quiet
assert.equal(linkHealth({ ...base, conn: 'open', lastEventAt: 10_000 - 25_000 }).commandsWork, true)

// the brake is NEVER disabled, only labelled
for (const v of [lost, off, un, recon]) {
  const p = estopPosture(v)
  assert.equal(p.degraded, true)
  assert.match(p.title, /still worth pressing/)
  assert.match(p.title, /power switch/)
}
assert.equal(estopPosture(live).degraded, false)
assert.match(estopPosture(live).title, /keyboard shortcut/)

console.log('linkHealth: all assertions passed')

// Q88 — the SAME transport failure, two different sentences. When the rejection is our own
// lapsed sign-in, "the server rejected this session" is technically true and practically
// misleading: it points at the backend, and the measured incident was 19.3 hours of hunting a
// camera bug for an expired token.
{
  const base = { conn: 'unauthorized', browserOnline: true, everOpen: true, peerCount: 2, now: 1000 }
  const generic = linkHealth(base)
  const lapsed = linkHealth({ ...base, sessionExpired: true })
  assert.equal(lapsed.kind, 'unauthorized', 'it is still the same transport verdict')
  assert.equal(lapsed.commandsWork, false)
  assert.match(lapsed.headline, /sign-in has expired/i)
  assert.match(lapsed.detail, /Sign in again/)
  assert.match(lapsed.detail, /nothing is wrong with the robots/i)
  // Non-negotiable in either wording: a brake that cannot leave the browser must never be
  // implied to work, and the physical fallback must be named.
  assert.match(lapsed.detail, /STOP ALL cannot be sent/)
  assert.match(lapsed.detail, /power switch/)
  assert.match(lapsed.estopReason, /sign in again/i)
  // And the flag absent leaves the old sentence exactly as it was.
  assert.match(generic.headline, /server rejected this session/)
  assert.notEqual(generic.detail, lapsed.detail)
}
console.log('linkHealth: Q88 lapsed-sign-in wording ok')
