// Assertions for the local-address hint (lib/lanHint.ts).
// Run: npx esbuild src/lib/lanHint.ts --bundle --format=esm --outfile=/tmp/lanHint.mjs \
//        && node src/lib/lanHint.test.mjs
import assert from 'node:assert/strict'

const { lanHintVerdict, readDismissed, sameOrigin } = await import('/tmp/lanHint.mjs')

const LOCAL = { same_network: true, lan_urls: ['http://192.168.1.164:8090'], client_ip: '2600:4041:4256:7e00:a13b::1' }

{
  const v = lanHintVerdict({ body: LOCAL, origin: 'https://robots.cagatay.my', dismissed: [] })
  assert.equal(v.show, true)
  assert.equal(v.url, 'http://192.168.1.164:8090')
  assert.match(v.text, /same network/)
}

// A page ALREADY on the local address must never be told to go there.
{
  const v = lanHintVerdict({ body: LOCAL, origin: 'http://192.168.1.164:8090', dismissed: [] })
  assert.equal(v.show, false)
  assert.match(v.reason, /already served/)
}

// Same host, DIFFERENT port is somewhere else - do not treat it as already-there.
{
  const v = lanHintVerdict({ body: LOCAL, origin: 'http://192.168.1.164:3000', dismissed: [] })
  assert.equal(v.show, true)
}

// 'unknown' (IPv4 behind NAT) and 'false' both mean SILENCE. A wrong "you are local"
// sends the operator to an unreachable URL, which is worse than the wasted bandwidth.
for (const same of [null, undefined, false]) {
  const v = lanHintVerdict({ body: { same_network: same, lan_urls: ['http://192.168.1.164:8090'] }, origin: 'https://x.y', dismissed: [] })
  assert.equal(v.show, false, `same_network=${same} must not show`)
}

// Local but no address to offer: still silent - never invent one.
{
  const v = lanHintVerdict({ body: { same_network: true, lan_urls: [] }, origin: 'https://x.y', dismissed: [] })
  assert.equal(v.show, false)
  assert.match(v.reason, /named no address/)
}

// An OLD server 404s this endpoint, so the caller passes null: silence, not a crash.
assert.equal(lanHintVerdict({ body: null, origin: 'https://x.y', dismissed: [] }).show, false)

// Dismissal is per URL: dismissing one address does not silence a different one.
{
  const dismissed = ['http://192.168.1.164:8090']
  assert.equal(lanHintVerdict({ body: LOCAL, origin: 'https://x.y', dismissed }).show, false)
  const other = { same_network: true, lan_urls: ['http://10.0.0.5:8090'] }
  assert.equal(lanHintVerdict({ body: other, origin: 'https://x.y', dismissed }).show, true)
}

// Only http:// LAN candidates are trusted; junk in the list cannot become the offer.
{
  const junk = { same_network: true, lan_urls: ['javascript:alert(1)', 'not a url', 'http://192.168.1.9:8090'] }
  const v = lanHintVerdict({ body: junk, origin: 'https://x.y', dismissed: [] })
  assert.equal(v.url, 'http://192.168.1.9:8090')
}

// A corrupt localStorage entry must not hide a working hint forever.
assert.deepEqual(readDismissed({ getItem: () => '{{{' }), [])
assert.deepEqual(readDismissed({ getItem: () => '"nope"' }), [])
assert.deepEqual(readDismissed(null), [])
assert.deepEqual(readDismissed({ getItem: () => '["http://a"]' }), ['http://a'])

assert.equal(sameOrigin('http://1.2.3.4:80', 'http://1.2.3.4'), true)
assert.equal(sameOrigin('nonsense', 'http://1.2.3.4'), false)

console.log('lanHint: all assertions passed')

// --- handoffHref: the sign-in rides the link, and every failure is the plain link
{
  const { handoffHref } = await import('/tmp/lanHint.mjs')
  const url = 'http://192.168.1.166:8090'
  // a minted token rides in ?token= (AuthGate's absorbUrl reads exactly this)
  assert.equal(handoffHref(url, { token: 'abc.def.ghi' }), 'http://192.168.1.166:8090/?token=abc.def.ghi')
  // auth off (token:null), old server (no body), refusal (undefined) — all degrade to the plain link
  assert.equal(handoffHref(url, { token: null, why: 'auth is not enabled' }), url)
  assert.equal(handoffHref(url, null), url)
  assert.equal(handoffHref(url, undefined), url)
  assert.equal(handoffHref(url, {}), url)
  // whitespace is not a token
  assert.equal(handoffHref(url, { token: '  ' }), url)
  // a malformed candidate must not become a malformed navigation
  assert.equal(handoffHref('not a url', { token: 'abc' }), 'not a url')
  console.log('handoffHref: all assertions passed')
}
