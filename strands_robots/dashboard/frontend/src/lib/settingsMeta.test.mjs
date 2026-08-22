// Run: npx esbuild src/lib/settingsMeta.ts --bundle --format=esm
// --outfile=/tmp/settingsMeta.mjs && node src/lib/settingsMeta.test.mjs Every tunable in the
// Settings drawer is validated here, and a validator that DISAGREES WITH THE VALUE'S CONSUMER
// is worse than none, because it is believed.
import assert from 'node:assert/strict'
import {
  SETTINGS, SEARCH_INDEX, APPLY_LABEL, settingMeta, validateSetting, allValid,
  finiteNumber, connectEndpoints, listenEndpoints, envKeyError, envValueError, searchSettings,
} from '/tmp/settingsMeta.mjs'

// ── the table itself ──
const keys = SETTINGS.map(s => s.key)
assert.equal(new Set(keys).size, keys.length, 'no key is described twice')
for (const s of SETTINGS) {
  assert.ok(s.label && s.effect, `${s.key} explains itself`)
  assert.ok(APPLY_LABEL[s.apply], `${s.key} has a real apply mode`)
  // the safe default must PASS its own validator, or the drawer offers a value it then refuses
  assert.equal(s.validate(s.safeDefault), null, `${s.key} safeDefault "${s.safeDefault}" is acceptable`)
  assert.equal(s.validate(''), null, `${s.key}: empty means "use the default" and is always valid`)
}
assert.ok(settingMeta('mesh.port'))
assert.equal(settingMeta('nope.nope'), undefined)
// an unknown key never blocks a save on missing metadata
assert.equal(validateSetting('nope.nope', 'anything'), null)
assert.equal(validateSetting('mesh.port', '99999'), 'maximum is 65535')
assert.equal(allValid({ 'mesh.port': '7447', 'agent.temperature': '0.7' }), true)
assert.equal(allValid({ 'mesh.port': 'seven' }), false)

assert.equal(finiteNumber(''), null)
assert.equal(finiteNumber('  '), null)
assert.equal(finiteNumber('1.5'), null)
assert.match(finiteNumber('abc'), /must be a number/)
assert.match(finiteNumber('NaN'), /must be a number/)
assert.match(finiteNumber('Infinity'), /must be a number/)
assert.match(finiteNumber('1.5', { integer: true }), /whole number/)
assert.match(finiteNumber('-1', { min: 0 }), /minimum is 0/)
assert.match(finiteNumber('3', { max: 2 }), /maximum is 2/)
assert.equal(validateSetting('agent.temperature', '2'), null, 'the boundary is inclusive')
assert.match(validateSetting('agent.temperature', '2.1'), /maximum is 2/)
assert.match(validateSetting('agent.max_tokens', '0'), /minimum is 1/, 'a 0-token cap truncates every reply')
assert.match(validateSetting('mesh.camera_hz', '0'), /minimum is 0.1/)
assert.equal(validateSetting('mesh.camera_hz', '60'), null)
assert.match(validateSetting('mesh.camera_hz', '61'), /maximum is 60/)

// ── endpoints: the schemes the MESH accepts, in both directions ──
// TLS-bearing schemes are valid under either posture.
for (const ep of ['tls/robot.lan:7447', 'quic/10.0.0.4:7447', 'wss/robot.lan:443', 'unixsock/tmp/zenoh.sock']) {
  assert.equal(connectEndpoints(ep), null, `${ep} is accepted`)
  assert.equal(listenEndpoints(ep), null, `${ep} is accepted to listen on`)
}
// wss and unixsock are exactly what the old regex refused, and they are the only extra transports a
// locked-down desk has.
assert.equal(listenEndpoints('wss/0.0.0.0:443'), null)
// tcp/udp are legal shapes that the DEFAULT posture refuses at mesh restart — say so at typing time
const plain = connectEndpoints('tcp/10.0.0.4:7447')
assert.match(plain, /default mTLS posture refuses/)
assert.match(plain, /STRANDS_MESH_AUTH_MODE=none/, 'and names the escape hatch')
assert.match(listenEndpoints('udp/0.0.0.0:7447'), /mTLS posture refuses/)
// a scheme the mesh has never heard of is a different (harder) error
assert.match(connectEndpoints('http/robot.lan:80'), /not a mesh transport/)
assert.match(connectEndpoints('robot.lan:7447'), /not proto\/host:port/, 'a bare host:port is not an endpoint')

assert.equal(listenEndpoints('tls/127.0.0.1:0'), null, 'the UI can express what session.py does')
assert.match(connectEndpoints('tls/127.0.0.1:0'), /cannot be dialled/)
assert.match(connectEndpoints('tls/robot.lan:70000'), /out-of-range/)
assert.match(connectEndpoints('tls/robot.lan:port'), /non-numeric port/)
assert.match(connectEndpoints('tls/robot.lan'), /missing a port/)
assert.match(connectEndpoints('unixsock/'), /needs a socket path/)
// zenoh's per-endpoint config suffix must survive validation
assert.equal(listenEndpoints('tls/0.0.0.0:7447#iface=en0'), null)
// a list, with whitespace and a trailing comma, and the FIRST bad entry names itself
assert.equal(connectEndpoints(' tls/a.lan:7447 , quic/b.lan:7447 , '), null)
assert.match(connectEndpoints('tls/a.lan:7447,http/b.lan:80'), /"http"/)

assert.equal(envKeyError(''), null)
assert.equal(envKeyError('HF_TOKEN'), null)
assert.equal(envKeyError('A1_B'), null)
assert.match(envKeyError('hf_token'), /UPPER_SNAKE_CASE/)
assert.match(envKeyError('1TOKEN'), /not starting with a digit/)
assert.match(envKeyError('MY TOKEN'), /UPPER_SNAKE_CASE/)
assert.match(envKeyError('A\nB=x'), /single line/, 'a newline would write a second variable')
assert.match(envKeyError('A\r\nB'), /single line/)
assert.equal(envValueError('hf_abc123'), null)
assert.equal(envValueError(''), null)
assert.equal(envValueError('a value with spaces and = signs'), null)
assert.match(envValueError('one\ntwo'), /second variable/)
assert.match(envValueError('one\rtwo'), /single line/)

// ── search: find a field without knowing which tab hides it ──
assert.deepEqual(searchSettings(''), [])
assert.deepEqual(searchSettings('   '), [])
const fps = searchSettings('fps')
assert.equal(fps[0].key, 'mesh.camera_hz', 'the word a user actually types for camera rate')
assert.equal(searchSettings('temperature')[0].key, 'agent.temperature')
assert.equal(searchSettings('TEMPERATURE')[0].key, 'agent.temperature', 'case-insensitive')
// every term must match, so more words NARROW
assert.ok(searchSettings('camera rate').length >= 1)
assert.equal(searchSettings('camera zzzz').length, 0)
// a label prefix outranks a mere mention
assert.equal(searchSettings('voice')[0].label.toLowerCase().startsWith('voice'), true)
assert.ok(searchSettings('token').length >= 2, 'both the browser token and the server token are findable')
assert.ok(searchSettings('a', 3).length <= 3, 'the limit is honoured')
// every indexed entry lands on a real tab, and no key is indexed twice with two explanations
const ikeys = SEARCH_INDEX.map(e => e.key)
assert.equal(new Set(ikeys).size, ikeys.length, 'no duplicate search entry (voice used to be listed twice)')
for (const e of SEARCH_INDEX) {
  assert.ok(['connection', 'agent', 'voice', 'mesh', 'env', 'security'].includes(e.tab), `${e.key} tab`)
  assert.ok(e.label && e.effect, `${e.key} explains itself in search too`)
  const meta = settingMeta(e.key)
  // where a field has metadata, the search result must quote the FIELD's own explanation
  if (meta) assert.equal(e.effect, meta.effect, `${e.key}: search text matches the drawer's text`)
}

console.log('settingsMeta: all assertions passed')
