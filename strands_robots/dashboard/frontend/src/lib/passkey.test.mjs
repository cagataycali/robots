// Run: node scripts/run-lib-tests.mjs passkey passkey.ts is the ONLY door into this dashboard
// from outside the LAN, and it had no test.
import assert from 'node:assert/strict'

const store = new Map()
globalThis.localStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: k => store.delete(k),
}
globalThis.location = { search: '', host: 'dash.local', origin: 'https://dash.local' }

const order = []
let credGet = async () => ({ id: 'cred-1', rawId: new Uint8Array([1, 2]), type: 'public-key', response: {} })
const defineGlobal = (name, value) =>
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true })

defineGlobal('navigator', {
  credentials: {
    create: async () => ({}),
    get: async opts => { order.push('credentials.get'); return credGet(opts) },
  },
})
defineGlobal('window', { isSecureContext: true, PublicKeyCredential: function () {} })

globalThis.fetch = async (url, init) => {
  order.push(`fetch ${new URL(String(url), 'https://dash.local').pathname}`)
  const body = String(url).includes('/openapi.json')
    ? { paths: { '/api/auth/login/begin': {}, '/api/auth/login/finish': {} } }
    : { challenge_id: 'chal-1', options: { challenge: 'AAAA', allowCredentials: [{ id: 'AAAA', type: 'public-key' }] }, token: 'sess-tok' }
  return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } })
}

const mod = await import('/tmp/passkey.mjs')

// ── 1. the wire format survives a round trip, including the url-safe alphabet ──
const bytes = new Uint8Array([0, 62, 63, 251, 252, 253, 255])
const b64u = mod.bufToB64u(bytes)
assert.ok(!/[+/=]/.test(b64u), `base64url must not contain + / or =, got ${b64u}`)
assert.deepEqual(new Uint8Array(mod.b64uToBuf(b64u)), bytes, 'buf -> b64u -> buf is lossless')
assert.deepEqual(new Uint8Array(mod.b64uToBuf('AAAA')), new Uint8Array([0, 0, 0]), 'unpadded input decodes')
assert.deepEqual(new Uint8Array(mod.b64uToBuf(mod.bufToB64u(new Uint8Array([1])))), new Uint8Array([1]),
                 'a 1-byte value (2 chars + 2 pad) decodes — the padding maths is the easy thing to get wrong')

// ── 2. THE iOS WEDGE FIX: prepGet hands the authenticator NO credential list ──
const getOpts = mod.prepGet({ challenge: 'AAAA', rpId: 'dash.local', allowCredentials: [{ id: 'AAAA', type: 'public-key' }] })
assert.ok(!('allowCredentials' in getOpts),
          'prepGet MUST drop allowCredentials: with a list and no transports hint iOS Safari never opens the sheet')
assert.equal(getOpts.rpId, 'dash.local', 'everything else is passed through untouched')
assert.deepEqual(new Uint8Array(getOpts.challenge), new Uint8Array([0, 0, 0]), 'challenge is decoded to bytes')

// prepCreate is the opposite case: excludeCredentials is a SERVER instruction ("this device
// already has one"), not a picker hint, so it is decoded and kept.
const createOpts = mod.prepCreate({
  challenge: 'AAAA', user: { id: 'AQID', name: 'c' }, excludeCredentials: [{ id: 'AAAA', type: 'public-key' }],
})
assert.equal(createOpts.excludeCredentials.length, 1, 'prepCreate KEEPS excludeCredentials')
assert.deepEqual(new Uint8Array(createOpts.excludeCredentials[0].id), new Uint8Array([0, 0, 0]), 'and decodes its id')
assert.deepEqual(new Uint8Array(createOpts.user.id), new Uint8Array([1, 2, 3]), 'user.id is decoded')

// ── 3. credToJSON: two ceremonies, two payload shapes ──
const attestation = mod.credToJSON({
  id: 'c1', rawId: new Uint8Array([1]), type: 'public-key',
  response: { attestationObject: new Uint8Array([2]), clientDataJSON: new Uint8Array([3]) },
})
assert.deepEqual(Object.keys(attestation.response).sort(), ['attestationObject', 'clientDataJSON'],
                 'enroll sends the attestation shape and nothing else')
const assertion = mod.credToJSON({
  id: 'c1', rawId: new Uint8Array([1]), type: 'public-key',
  response: { authenticatorData: new Uint8Array([2]), clientDataJSON: new Uint8Array([3]), signature: new Uint8Array([4]) },
})
assert.deepEqual(Object.keys(assertion.response).sort(), ['authenticatorData', 'clientDataJSON', 'signature', 'userHandle'],
                 'login sends the assertion shape')
assert.equal(assertion.response.userHandle, null,
             'an absent userHandle is explicit null — undefined would vanish from the JSON body')
assert.deepEqual(attestation.clientExtensionResults, {},
                 'a credential without getClientExtensionResults still produces the key the server reads')

// ── 4. freshness: a challenge the server has already expired must not be tapped ──
assert.equal(mod.loginFresh(null), false, 'no prepared login is not fresh')
assert.equal(mod.loginFresh({ challenge_id: 'c', options: {}, t: Date.now() }), true, 'just-fetched is fresh')
assert.equal(mod.loginFresh({ challenge_id: 'c', options: {}, t: Date.now() - 241_000 }), false,
             'past 240s it is refetched — the server drops challenges at 300s')

// ── 5. webauthnReady gates on the SECURE CONTEXT, not on user agent sniffing ──
assert.equal(mod.webauthnReady(), true, 'https + credentials + PublicKeyCredential = ready')
globalThis.window.isSecureContext = false
assert.equal(mod.webauthnReady(), false, 'plain http://<lan-ip> is answered up front, not by crashing in .create()')
globalThis.window.isSecureContext = true

// ── 6. THE USER-ACTIVATION LAW: completeLogin touches the authenticator before the network ──
order.length = 0
const token = await mod.completeLogin({ challenge_id: 'chal-1', options: { challenge: 'AAAA' }, t: Date.now() })
assert.equal(token, 'sess-tok', 'the session token is returned')
assert.equal(order[0], 'credentials.get',
             `credentials.get MUST be the first await — any fetch before it spends the tap's user activation and ` +
             `Safari then refuses the ceremony silently. Order was: ${order.join(' -> ')}`)
assert.ok(order.includes('fetch /api/auth/login/finish'), 'the credential is then verified server-side')

// beginLogin drops the list too — prepGet alone is not enough, because the gate persists a PreparedLogin
// and other code paths read p.options directly.
order.length = 0
const prepared = await mod.beginLogin()
assert.ok(!('allowCredentials' in prepared.options),
          'beginLogin strips allowCredentials from the STORED options, not just at ceremony time')
assert.ok(prepared.t > 0 && mod.loginFresh(prepared), 'it is stamped so loginFresh can judge it')

// ── 7. a silent authenticator ends with an instruction, not a spinner ── The wedge itself:
// the sheet never resolves.
credGet = opts => new Promise((_, rej) => {
  opts.signal.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError')))
})
await assert.rejects(
  mod.completeLogin({ challenge_id: 'chal-1', options: { challenge: 'AAAA' }, t: Date.now() }, 20),
  /did not answer in time — tap sign in to try again/,
  'the abort is reported as something the human can DO; Safari has been seen ignoring options.timeout')

// A cancelled sheet is a different sentence: nothing is wrong with the site.
credGet = async () => null
await assert.rejects(
  mod.completeLogin({ challenge_id: 'chal-1', options: { challenge: 'AAAA' }, t: Date.now() }),
  /passkey sign-in was cancelled/, 'a null credential means the user dismissed it')

{  // sections 8-10 keep their fixtures in a block: the names above are the sibling test's
  // ── 8. base64url: the substitution PROVEN by the two bytes that need it ── Section 1 asserts
  // no + / = survive; this asserts the mapping is right, which a payload of alphanumerics can
  // never catch. 0xFB 0xFF encodes to "+/" in the standard alphabet, so if the replace() pair is
  // ever dropped this is the input that says so.
  assert.equal(mod.bufToB64u(new Uint8Array([0xfb, 0xff, 0x00])), '-_8A', 'base64url must use -_ and drop padding')
  for (const n of [1, 2, 3, 16, 32, 64]) {  // every length remainder, i.e. every padding case
    const bs = new Uint8Array(Array.from({ length: n }, (_, i) => (i * 37 + 11) & 0xff))
    assert.deepEqual(new Uint8Array(mod.b64uToBuf(mod.bufToB64u(bs))), bs, `round trip at ${n} bytes`)
  }
  // A padded, standard-alphabet string is what a curl probe or an older server sends; rejecting
  // it would surface as a corrupt credential.
  assert.deepEqual(new Uint8Array(mod.b64uToBuf('+/8=')), new Uint8Array([0xfb, 0xff]))

  // ── 9. the prep functions do not mutate the caller's options ── The gate persists a
  // PreparedLogin and re-reads p.options; if prep* edited it in place, a second ceremony would
  // run against half-converted options (bytes where the server's base64url is expected).
  const getOpts = { challenge: 'AAEC', rpId: 'robots.cagatay.my', allowCredentials: [{ id: 'AAEC', type: 'public-key' }] }
  const gotGet = mod.prepGet(getOpts)
  assert.ok(Array.isArray(getOpts.allowCredentials), 'prepGet must not strip the CALLER\'s list')
  assert.equal(gotGet.rpId, 'robots.cagatay.my', 'and everything it does not convert survives')
  const createOpts = { challenge: 'AAEC', user: { id: '-_8A', name: 'c' }, excludeCredentials: [{ id: 'AAEC', type: 'public-key' }] }
  const gotCreate = mod.prepCreate(createOpts)
  assert.equal(typeof createOpts.user.id, 'string', 'prepCreate must not convert the caller\'s user handle in place')
  assert.deepEqual(new Uint8Array(gotCreate.user.id), new Uint8Array([0xfb, 0xff, 0x00]),
                   'the user handle is the field most likely to carry -_, and it must reach the browser as bytes')
  assert.deepEqual(new Uint8Array(gotCreate.excludeCredentials[0].id), new Uint8Array([0, 1, 2]),
                   'excludeCredentials ids are bytes too, or re-enrolment silently duplicates a key')
  assert.equal(gotCreate.excludeCredentials[0].type, 'public-key', 'the rest of the entry survives')
  // Absent and [] are DIFFERENT instructions to an authenticator, so an enrolment with nothing to
  // exclude must not grow the field.
  assert.ok(!('excludeCredentials' in mod.prepCreate({ challenge: 'AAEC', user: { id: 'AAEC' } })))

  // ── 10. the freshness window's exact boundary ── Section 4 pins the principle; these pin the
  // number, because the risk is a widening that still "looks fresh": anything at or past the
  // server's 300s TTL must be refused.
  const t0 = Date.now()
  assert.equal(mod.loginFresh({ challenge_id: 'c', options: {}, t: t0 - 239_000 }), true, 'just inside')
  assert.equal(mod.loginFresh({ challenge_id: 'c', options: {}, t: t0 - 241_000 }), false, 'just outside')
  assert.equal(mod.loginFresh({ challenge_id: 'c', options: {}, t: t0 - 299_000 }), false,
               'a challenge the server may already have expired must never count as fresh')
}

console.log('passkey.test.mjs: all assertions passed (sections 1-10)')
