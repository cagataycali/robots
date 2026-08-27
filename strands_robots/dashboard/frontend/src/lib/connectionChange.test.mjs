import assert from 'node:assert/strict'
const { connectionChange, needsConfirm } = await import('/tmp/connectionChange.mjs')

const P = { pageHost: 'robots.cagatay.my' }

// THE BUG: change only the URL, leave the password box alone -> robot A's secret is sent to robot B.
{
  const v = connectionChange({ ...P, currentBase: 'https://a.lan:8090', currentToken: 'S', nextBase: 'https://b.lan:8090', nextToken: 'S' })
  assert.equal(v.kind, 'token_follows_host')
  assert.equal(needsConfirm(v), true)
  assert.equal(v.fromHost, 'a.lan:8090')
  assert.equal(v.toHost, 'b.lan:8090')
  assert.match(v.detail, /given for a\.lan:8090/)
  assert.match(v.detail, /the secret is gone/)
  // The alternative must exist: a warning with only "OK" is a dead end.
  assert.match(v.alternative, /without a token/)
}

// Same host, same token: the ordinary save. MUST be silent - a dialog on every harmless save teaches
// people to click through the one that mattered.
assert.deepEqual(connectionChange({ ...P, currentBase: 'https://a.lan', currentToken: 'S', nextBase: 'https://a.lan', nextToken: 'S' }), { kind: 'ok' })
// Port is part of the identity: :8090 and :9000 are different servers.
assert.equal(connectionChange({ ...P, currentBase: 'https://a.lan:8090', currentToken: 'S', nextBase: 'https://a.lan:9000', nextToken: 'S' }).kind, 'token_follows_host')
// Scheme alone is not a host change.
assert.equal(connectionChange({ ...P, currentBase: 'http://a.lan:8090', currentToken: 'S', nextBase: 'https://a.lan:8090', nextToken: 'S' }).kind, 'ok')

// A token the operator JUST TYPED is theirs to aim anywhere - they are looking at it as they press.
assert.equal(connectionChange({ ...P, currentBase: 'https://a.lan', currentToken: 'S', nextBase: 'https://b.lan', nextToken: 'NEW' }).kind, 'ok')
// Clearing the token can never leak it.
assert.equal(connectionChange({ ...P, currentBase: 'https://a.lan', currentToken: 'S', nextBase: 'https://b.lan', nextToken: '' }).kind, 'ok')

// Empty base means "the origin that served this page", so leaving a robot's URL for the origin is
// still a host change - and the wording says which one in words, not as an empty string.
{
  const v = connectionChange({ ...P, currentBase: 'https://a.lan', currentToken: 'S', nextBase: '', nextToken: 'S' })
  assert.equal(v.kind, 'token_follows_host')
  assert.equal(v.toHost, 'robots.cagatay.my')
  const back = connectionChange({ ...P, currentBase: '', currentToken: 'S', nextBase: 'https://a.lan', nextToken: 'S' })
  assert.equal(back.fromHost, 'robots.cagatay.my')
}

// An unparseable base is refused BEFORE the reload: afterwards every request fails against a URL the
// operator can no longer see, which reads like a dead backend.
{
  const v = connectionChange({ ...P, currentBase: '', currentToken: '', nextBase: 'http://', nextToken: '' })
  assert.equal(v.kind, 'unparseable')
  assert.match(v.detail, /host:port or a full URL/)
  assert.equal(needsConfirm(v), false, 'a refusal is not a confirmation')
}
// A bare host:port is normal typing, not garbage.
assert.equal(connectionChange({ ...P, currentBase: '', currentToken: '', nextBase: 'robot.lan:8090', nextToken: '' }).kind, 'ok')

// Clear text: the token crosses the wire readable. Only when it actually leaves this machine.
{
  const v = connectionChange({ ...P, currentBase: '', currentToken: '', nextBase: 'http://robot.lan:8090', nextToken: 'S' })
  assert.equal(v.kind, 'cleartext_token')
  assert.match(v.detail, /clear text/)
  assert.match(v.detail, /move motors/)
  // A bare host:port defaults to http, so it carries the same risk and must not be treated as safe.
  assert.equal(connectionChange({ ...P, currentBase: '', currentToken: '', nextBase: 'robot.lan:8090', nextToken: 'S' }).kind, 'cleartext_token')
}
for (const local of ['http://localhost:8090', 'http://127.0.0.1:8090', 'http://[::1]:8090']) {
  assert.equal(connectionChange({ ...P, currentBase: '', currentToken: '', nextBase: local, nextToken: 'S' }).kind, 'ok', local)
}
// https is the point of the advice, so it must not warn.
assert.equal(connectionChange({ ...P, currentBase: '', currentToken: '', nextBase: 'https://robot.lan', nextToken: 'S' }).kind, 'ok')

// Two problems at once: the one that loses the secret to the WRONG PARTY outranks the one that exposes
// it on the path.
assert.equal(connectionChange({ ...P, currentBase: 'https://a.lan', currentToken: 'S', nextBase: 'http://b.lan', nextToken: 'S' }).kind, 'token_follows_host')

console.log('connectionChange: all assertions passed')
