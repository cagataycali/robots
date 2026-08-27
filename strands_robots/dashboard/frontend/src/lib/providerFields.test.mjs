// build: npx esbuild src/lib/providerFields.ts --bundle --format=esm --outfile=/tmp/pf.mjs
import assert from 'node:assert/strict'
const { extraFields, missingForProvider } = await import('/tmp/pf.mjs')

// groot gets exactly the field its validate() demands, and it is a real spec key
{
  const f = extraFields('groot')
  assert.equal(f.length, 1)
  assert.equal(f[0].key, 'embodiment')
  assert.match(f[0].say, /embodiment_tag/, 'quote the flag the trainer names, so the error matches the field')
  assert.equal(f[0].required, true)
}

// nobody else gets it: an input that does nothing teaches people to ignore inputs
for (const p of ['lerobot_local', 'mock', 'cosmos3', 'ppo', 'wat']) {
  assert.deepEqual(extraFields(p), [], `${p} must not grow a GR00T field`)
}

// the form can say what is missing before the server refuses
assert.match(missingForProvider('groot', {}), /embodiment/)
assert.match(missingForProvider('groot', { embodiment: '   ' }), /embodiment/, 'whitespace is not an answer')
assert.equal(missingForProvider('groot', { embodiment: 'new_embodiment' }), '')
assert.equal(missingForProvider('lerobot_local', {}), '', 'silence when nothing is missing')

console.log('providerFields: ok')
