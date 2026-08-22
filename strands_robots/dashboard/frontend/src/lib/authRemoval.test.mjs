import assert from 'node:assert/strict'
const { authRemovalWarning, isLocalHost } = await import('/tmp/authRemoval.mjs')

// Browsing over a real hostname PROVES off-box reachability - the operator is doing it right now, so
// the warning states it as fact instead of hedging.
{
  const w = authRemovalWarning({ host: 'robots.cagatay.my', peerCount: 3 })
  assert.equal(w.severity, 'exposed')
  assert.match(w.lines[0], /robots\.cagatay\.my/)
  assert.match(w.lines[0], /reachable from outside/)
  // The confirm label must carry the consequence, not say "OK".
  assert.match(w.confirmLabel, /open to the network/)
}

// Localhost is not safe, only unproven: the network and any tunnel still reach the port.
{
  for (const host of ['localhost', '127.0.0.1', '::1', '[::1]', '0.0.0.0', '']) {
    assert.equal(isLocalHost(host), true, host)
    const w = authRemovalWarning({ host })
    assert.equal(w.severity, 'local')
    assert.match(w.lines[0], /another machine on this network|tunnel/)
  }
  assert.equal(isLocalHost('robots.cagatay.my'), false)
  assert.equal(isLocalHost('LOCALHOST'), true, 'case must not decide a security warning')
}

// The robots are counted, because "3 robots can be commanded" lands where "unauthenticated" does not.
assert.match(authRemovalWarning({ host: 'x.lan', peerCount: 3 }).lines[1], /3 robots/)
assert.match(authRemovalWarning({ host: 'x.lan', peerCount: 1 }).lines[1], /^1 robot on this fleet/, 'singular')
// An empty fleet is not an argument that it is safe: the next robot to join inherits the exposure.
assert.match(authRemovalWarning({ host: 'x.lan', peerCount: 0 }).lines[1], /Any robot that joins/)
assert.match(authRemovalWarning({ host: 'x.lan' }).lines[1], /Any robot that joins/)

// CORS '*' adds a distinct reach (a page on any site), so it gets its own line - only when set.
assert.ok(authRemovalWarning({ host: 'x.lan', corsOrigins: '*' }).lines.some(l => /any site/.test(l)))
assert.ok(!authRemovalWarning({ host: 'x.lan', corsOrigins: 'https://a.example' }).lines.some(l => /any site/.test(l)))
assert.ok(!authRemovalWarning({ host: 'x.lan', corsOrigins: null }).lines.some(l => /any site/.test(l)))

// Always tell them the way back: an operator who can see the undo is likelier to re-lock it.
for (const facts of [{ host: 'x.lan' }, { host: 'localhost' }]) {
  assert.ok(authRemovalWarning(facts).lines.some(l => /re-locks it immediately/.test(l)), JSON.stringify(facts))
}

console.log('authRemoval: all assertions passed')
