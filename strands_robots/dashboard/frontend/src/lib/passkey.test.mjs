/**
 * lib/passkey.ts — the base64url + options plumbing every sign-in passes through.
 *
 * These are the functions behind two real incidents, so the tests are written to fail if either
 * regression is reintroduced rather than to cover lines:
 *   · iOS Safari never opened the Face ID sheet while allowCredentials carried an entry with no
 *     transports hint (2026-08-19, cost an evening of "waiting for authenticator"). prepGet DELETES
 *     allowCredentials on purpose — passkeys are discoverable, and the server verifies the returned
 *     credential id against its store anyway. A future "let's be explicit" refactor must break a test.
 *   · a challenge fetched on demand raced the tap, so the gate pre-fetches one and reuses it while
 *     fresh. loginFresh owns that window (server keeps challenges 300s, we trust 240s).
 * And base64url itself: WebAuthn fails as "invalid credential", never as "your padding is wrong", so
 * the round trip is pinned including the byte values that PROVE the -_ / +/ substitution happened.
 *
 * Run: npx esbuild src/lib/passkey.ts --bundle --format=esm --outfile=/tmp/passkey.mjs && node src/lib/passkey.test.mjs
 */
import assert from 'node:assert/strict'

const m = await import('/tmp/passkey.mjs')

// ── base64url ──────────────────────────────────────────────────────────────────────────────────────
{
  // 0xFB 0xFF encodes to "+/" in standard base64 — the two characters base64url must replace. If the
  // substitution is dropped, this is the input that catches it (a-zA-Z0-9 payloads never would).
  const bytes = new Uint8Array([0xfb, 0xff, 0x00])
  const s = m.bufToB64u(bytes)
  assert.equal(s, '-_8A', 'base64url must use -_ and drop padding')
  assert.deepEqual(new Uint8Array(m.b64uToBuf(s)), bytes, 'round trip')
}
{
  // Every length class: the server sends unpadded base64url, so b64uToBuf must re-pad. A 1-byte and a
  // 2-byte payload are the two remainders that need padding at all.
  for (const n of [1, 2, 3, 16, 32, 64]) {
    const bytes = new Uint8Array(Array.from({ length: n }, (_, i) => (i * 37 + 11) & 0xff))
    const back = new Uint8Array(m.b64uToBuf(m.bufToB64u(bytes)))
    assert.deepEqual(back, bytes, `round trip at ${n} bytes`)
    assert.ok(!m.bufToB64u(bytes).includes('='), 'no padding on the wire')
  }
}
{
  // A padded, standard-alphabet string is what a hand-written curl test or an older server sends.
  // Accepting it costs nothing and its rejection would look like a corrupt credential.
  // (Hand-computing this expectation wrong is how I learned the test was reading the real decoder:
  //  '+/8=' is 0xFB 0xFF, the same two bytes as '-_8A' minus its trailing zero.)
  assert.deepEqual(new Uint8Array(m.b64uToBuf('+/8=')), new Uint8Array([0xfb, 0xff]))
}

// ── prepGet: the iOS incident ──────────────────────────────────────────────────────────────────────
{
  const opts = {
    challenge: 'AAEC', rpId: 'robots.cagatay.my', timeout: 60000,
    userVerification: 'required',
    allowCredentials: [{ id: 'AAEC', type: 'public-key' }],
  }
  const out = m.prepGet(opts)
  assert.ok(!('allowCredentials' in out),
    'prepGet must NOT pin a credential list: iOS Safari never opened the sheet for an entry with no transports hint')
  assert.deepEqual(new Uint8Array(out.challenge), new Uint8Array([0, 1, 2]), 'challenge becomes bytes')
  assert.equal(out.rpId, 'robots.cagatay.my', 'everything else survives untouched')
  assert.equal(out.userVerification, 'required')
  assert.ok(Array.isArray(opts.allowCredentials), 'the caller\'s object is not mutated')
}

// ── prepCreate ─────────────────────────────────────────────────────────────────────────────────────
{
  const opts = {
    challenge: 'AAEC', rp: { id: 'x', name: 'x' },
    user: { id: '-_8A', name: 'cagatay', displayName: 'cagatay' },
    excludeCredentials: [{ id: 'AAEC', type: 'public-key' }],
  }
  const out = m.prepCreate(opts)
  assert.deepEqual(new Uint8Array(out.user.id), new Uint8Array([0xfb, 0xff, 0x00]),
    'the user handle is bytes, and it is the field most likely to carry -_')
  assert.deepEqual(new Uint8Array(out.excludeCredentials[0].id), new Uint8Array([0, 1, 2]),
    'excludeCredentials ids are bytes too, or re-enrolment silently duplicates a key')
  assert.equal(out.excludeCredentials[0].type, 'public-key', 'the rest of the entry survives')
  assert.equal(typeof opts.user.id, 'string', 'the caller\'s object is not mutated')
  // An enrolment with nothing to exclude must not invent the field: an empty array is a DIFFERENT
  // instruction to the authenticator than its absence.
  assert.ok(!('excludeCredentials' in m.prepCreate({ challenge: 'AAEC', user: { id: 'AAEC' } })))
}

// ── loginFresh: the pre-fetched challenge window ───────────────────────────────────────────────────
{
  const now = Date.now()
  assert.equal(m.loginFresh(null), false, 'nothing pre-fetched is not fresh')
  assert.equal(m.loginFresh({ challenge_id: 'c', options: {}, t: now }), true)
  assert.equal(m.loginFresh({ challenge_id: 'c', options: {}, t: now - 239_000 }), true, 'just inside')
  assert.equal(m.loginFresh({ challenge_id: 'c', options: {}, t: now - 241_000 }), false, 'just outside')
  // The window must stay UNDER the server's 300s, or the page taps with a challenge the server has
  // already forgotten and the failure reads as a broken authenticator.
  assert.ok(m.loginFresh({ challenge_id: 'c', options: {}, t: now - 299_000 }) === false,
    'a challenge the server may have expired must never count as fresh')
}

console.log('passkey.test.mjs: base64url + prepGet/prepCreate + loginFresh ok')
