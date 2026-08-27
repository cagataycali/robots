// build: npx esbuild src/lib/startSnippet.ts --bundle --format=esm --outfile=/tmp/ss.mjs
import assert from 'node:assert/strict'
const { startSnippet } = await import('/tmp/ss.mjs')

// --- nothing detected: the placeholder stays, but says it is one ---
{
  const s = startSnippet([])
  assert.match(s.code, /\/dev\/ttyACM0/)
  assert.equal(s.real, false)
  assert.match(s.provenance, /is an example/, 'a placeholder that looks measured is the bug being fixed')
  assert.deepEqual(startSnippet(null).code, s.code, 'a failed lookup behaves like no board, not like a crash')
}

// --- a real board: its real port, and the family it last ran as ---
{
  const s = startSnippet([{ device: '/dev/cu.usbmodem5AB0181806', robot_name: 'so101' }])
  assert.match(s.code, /port="\/dev\/cu\.usbmodem5AB0181806"/)
  assert.doesNotMatch(s.code, /ttyACM0/, 'the linux example must be gone once we know the truth')
  assert.equal(s.real, true)
  assert.match(s.provenance, /detected on this machine right now/)
  assert.match(s.provenance, /last spawned as "so101"/)
}

// --- an unremembered board still gives a real port, and does not invent a family ---
{
  const s = startSnippet([{ device: '/dev/cu.usbmodemBBB1' }])
  assert.match(s.code, /Robot\("so101", mode="real", port="\/dev\/cu\.usbmodemBBB1"\)/)
  assert.doesNotMatch(s.provenance, /last spawned/, 'never claim a family we were not told')
}

// --- a blank robot_name must not produce Robot("") ---
{
  const s = startSnippet([{ device: '/dev/x', robot_name: '  ' }])
  assert.match(s.code, /Robot\("so101"\)/)
  assert.doesNotMatch(s.code, /Robot\(""\)/)
}

// --- the sim line is always there: it is the only line that works with no hardware at all ---
for (const arg of [[], [{ device: '/dev/x' }]]) {
  assert.match(startSnippet(arg).code, /# sim, no hardware needed/)
  assert.match(startSnippet(arg).code, /^from strands_robots import Robot$/m)
}

console.log('startSnippet: ok')
