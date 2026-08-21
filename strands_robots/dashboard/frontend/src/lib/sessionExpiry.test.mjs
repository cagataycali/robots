// Assertions for the sign-in verdict (lib/sessionExpiry.ts).
// Run: npx esbuild src/lib/sessionExpiry.ts --bundle --format=esm --outfile=/tmp/sessionExpiry.mjs \
//        && node src/lib/sessionExpiry.test.mjs
import assert from 'node:assert/strict'

const { sessionVerdict, tokenExpiry, humaniseSeconds, EXPIRING_SOON_S } =
  await import('/tmp/sessionExpiry.mjs')

const b64url = (obj) => Buffer.from(JSON.stringify(obj)).toString('base64')
  .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
const jwt = (claims) => `${b64url({ alg: 'HS256', typ: 'JWT' })}.${b64url(claims)}.sIgNaTuRe`

// THE MEASURED TOKEN (Q88): the real payload from the live log, expired 19.3 hours before the read.
{
  const real = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJyLVFOMm1iV2lIRXEtT1BUQVgtQk9Kb2xtc2ci'
    + 'LCJuYW1lIjoiY2FnYXRheSIsImlhdCI6MTc4NzEyMzA4MCwiZXhwIjoxNzg3MjA5NDgwfQ.W3L8czgAYPYQO-zzCc-6C-HEDGwliiChAZR7jZuScQE'
  assert.equal(tokenExpiry(real), 1787209480)
  const v = sessionVerdict(real, 1787278900)
  assert.equal(v.state, 'expired')
  assert.equal(v.refusesUntilSignIn, true)
  assert.match(v.text, /expired 19 hours ago/)  // one decimal only below 10h — '19.3' is false precision here
  // It must say the robots are fine: the tile's own wording blames the camera, and that sent the
  // operator hunting hardware for a lapsed cookie.
  assert.match(v.text, /Nothing is wrong with the robots/)
  assert.match(v.text, /sign in again/)
}

// SILENCE where a sentence would be a lie: no token at all is a supported way to run this
// dashboard (auth off on a LAN), and an opaque bootstrap token never lapses.
for (const token of [null, undefined, '', '   ']) {
  const v = sessionVerdict(token, 1000)
  assert.equal(v.state, 'none')
  assert.equal(v.text, null)
  assert.equal(v.refusesUntilSignIn, false)
}
for (const token of ['local-bootstrap-token', 'a.b', 'not.a.jwt', jwt({ sub: 'x' })]) {
  const v = sessionVerdict(token, 1000)
  assert.equal(v.state, 'opaque', `${token} must not be judged`)
  assert.equal(v.text, null)
  assert.equal(v.refusesUntilSignIn, false)
  assert.equal(v.expiresInS, null)
}
// A non-numeric exp is not an expiry. (A server that ever sends "exp": "soon" must not lock the UI.)
assert.equal(sessionVerdict(jwt({ exp: 'soon' }), 1000).state, 'opaque')
assert.equal(sessionVerdict(jwt({ exp: Number.NaN }), 1000).state, 'opaque')

// A live session says NOTHING — a permanent countdown badge would train the operator to ignore it.
{
  const v = sessionVerdict(jwt({ exp: 10_000 }), 1000)
  assert.equal(v.state, 'valid')
  assert.equal(v.text, null)
  assert.equal(v.expiresInS, 9000)
}

// The warning window: at the boundary it warns, one second earlier it does not.
{
  const now = 5000
  const edge = sessionVerdict(jwt({ exp: now + EXPIRING_SOON_S }), now)
  assert.equal(edge.state, 'expiring')
  assert.match(edge.text, /lapses in 5 minutes/)
  // It names the consequence that actually costs something: a recording refused half-way.
  assert.match(edge.text, /recording/)
  assert.equal(edge.refusesUntilSignIn, false, 'an expiring session still works — do not stop retrying')
  assert.equal(sessionVerdict(jwt({ exp: now + EXPIRING_SOON_S + 1 }), now).state, 'valid')
}

// The exact moment of lapse counts as expired, not as "0 seconds left and fine".
assert.equal(sessionVerdict(jwt({ exp: 5000 }), 5000).state, 'expired')

// Wording scales: seconds, minutes, hours — and never "0.9 hours" where "54 minutes" reads better.
assert.equal(humaniseSeconds(40), '40 seconds')
assert.equal(humaniseSeconds(-40), '40 seconds')
assert.equal(humaniseSeconds(240), '4 minutes')
assert.equal(humaniseSeconds(3240), '54 minutes')
assert.equal(humaniseSeconds(69_480), '19 hours')
assert.equal(humaniseSeconds(9_000), '2.5 hours')
// FOUND BY THE BROWSER AUDIT, not by this file: a whole number of hours must not print "4.0 hours".
assert.equal(humaniseSeconds(4 * 3600), '4 hours')
// Below 90 minutes the sentence stays in minutes, which reads better than "1.5 hours".
assert.equal(humaniseSeconds(3600), '60 minutes')
assert.equal(humaniseSeconds(5400), '1.5 hours')  // a decimal earns its place while the number is small
assert.equal(humaniseSeconds(200_000), '56 hours')

// Garbage that would throw in a naive decoder must not take the page down with it.
for (const token of ['!!!.!!!.!!!', 'ey.%%%%.zz', '..', jwt({}) + '.extra']) {
  assert.doesNotThrow(() => sessionVerdict(token, 1000))
}

console.log('sessionExpiry: 6 assertions groups ok — the measured 19.3h-expired token is named, and a '
  + 'missing/opaque/valid credential stays silent')
