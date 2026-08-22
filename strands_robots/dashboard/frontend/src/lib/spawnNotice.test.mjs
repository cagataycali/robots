// Assertions for the spawn warning notice (lib/spawnNotice.ts).
// Run: npx esbuild src/lib/spawnNotice.ts --bundle --format=esm --outfile=/tmp/spawnNotice.mjs \
//        && node src/lib/spawnNotice.test.mjs
import assert from 'node:assert/strict'

const { spawnNotice } = await import('/tmp/spawnNotice.mjs')

// The live case: the spawn SUCCEEDED (there is a pid) and the arm will still have no joints.
{
  const n = spawnNotice({
    peer_id: 'so101-leader', pid: 4242, mode: 'real',
    calibration_warning: "robot_id 'leader' has a calibration, but as a teleoperator: /x/leader.json.",
  })
  assert.equal(n.tone, 'warn', 'a warning, not an error - the child did start')
  assert.match(n.text, /teleoperator/)
}

// A clean spawn says nothing. Silence here is the common case and must stay silent.
{
  assert.equal(spawnNotice({ peer_id: 'so101-arm-1', pid: 1, mode: 'real' }), null)
}

// Nothing to read: no body at all, or a body that is not an object.
{
  for (const b of [null, undefined, '', 'ok', 42, true, []]) {
    assert.equal(spawnNotice(b), null, `${JSON.stringify(b)} carries no notice`)
  }
}

// A non-string warning is NOT rendered.
{
  for (const v of [true, 1, {}, ['a'], null]) {
    assert.equal(spawnNotice({ calibration_warning: v }), null, `${JSON.stringify(v)} is not a sentence`)
  }
}

// Whitespace is not a sentence either - an empty amber box would be a mystery, not a message.
{
  assert.equal(spawnNotice({ calibration_warning: '   \n\t ' }), null)
  assert.equal(spawnNotice({ calibration_warning: '  no calibration  ' }).text, 'no calibration',
    'and a real sentence is trimmed rather than rejected')
}

// A failed body can carry it too (a settled-then-dead spawn reports 200-with-error), and the
// two are independent: the error line and the calibration reason are different facts about the
// same attempt.
{
  const n = spawnNotice({ error: 'exited after 1.2s', calibration_warning: 'no calibration for "leader"' })
  assert.match(n.text, /leader/)
}

console.log('spawnNotice: all assertions passed')
