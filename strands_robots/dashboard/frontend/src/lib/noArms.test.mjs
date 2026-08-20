// Q44: assertions for the record screen's empty state (lib/noArms.ts).
// Run: npx esbuild src/lib/noArms.ts --bundle --format=esm --outfile=/tmp/noArms.mjs && node src/lib/noArms.test.mjs
import assert from 'node:assert/strict'

const { noArmsVerdict } = await import('/tmp/noArms.mjs')

// --- arms present: nothing to explain ---
assert.equal(noArmsVerdict(2, []), null)
assert.equal(noArmsVerdict(1, null), null)

// --- THE COMMON CASE: a restart. The arms are configured, just not running. ---
{
  const v = noArmsVerdict(0, [{ peer_id: 'so101-arm-1' }, { peer_id: 'so101-arm-2' }])
  assert.match(v.text, /so101-arm-1 and so101-arm-2/, 'the boards must be named, not counted')
  assert.match(v.text, /one click there/)
  assert.equal(v.offerDevices, true)
}

// --- a board whose bus is already busy is not a route out of "no arms" ---
{
  const v = noArmsVerdict(0, [{ peer_id: 'so101-arm-1', claimed: true }])
  assert.doesNotMatch(v.text, /one click/, 'offering a respawn on a claimed bus would fail or collide')
  assert.match(v.text, /logs/, 'a child that holds the bus but never announced itself is a log question')
  assert.match(v.text, /rather than spawning again/, 'and it must say why a respawn is the wrong move')
}

// --- mixed: one busy, one free. The free one is the route. ---
{
  const v = noArmsVerdict(0, [{ peer_id: 'busy-arm', claimed: true }, { peer_id: 'free-arm' }])
  assert.match(v.text, /remembers free-arm/)
  assert.doesNotMatch(v.text, /busy-arm/, 'a board that cannot be spawned must not be offered as one')
}

// --- first run: nothing configured. Same destination, different work. ---
{
  const v = noArmsVerdict(0, [])
  assert.match(v.text, /no board is configured yet/)
  assert.match(v.text, /plug an arm in/)
  assert.match(v.text, /remembered by its USB serial/, 'say that the effort is not repeated next time')
}

// --- the request failed: absence of evidence is not evidence of absence ---
{
  const v = noArmsVerdict(0, null)
  assert.match(v.text, /could not be reached/)
  assert.doesNotMatch(v.text, /no board is configured/, 'a failed lookup must not become a claim')
  assert.equal(v.offerDevices, true)
}

// --- Q45: `route` is the same words without the "no arms" prefix, for a screen whose heading
// already said it. Two screens, one vocabulary.
for (const arg of [[{ peer_id: 'a' }], [{ peer_id: 'a', claimed: true }], [], null]) {
  const v = noArmsVerdict(0, arg)
  assert.doesNotMatch(v.route, /no arms/, 'the route must not repeat the absence')
  assert.ok(v.text.endsWith(v.route), 'text and route must be the same words, not two phrasings')
  assert.match(v.text, /^no arms are on the mesh/)
}

console.log('noArms: ok')
