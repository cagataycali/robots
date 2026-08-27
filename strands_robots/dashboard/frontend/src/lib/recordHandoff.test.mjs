// The record → train handoff rule (lib/recordHandoff.ts).
// Run: npx esbuild src/lib/recordHandoff.ts --bundle --format=esm --outfile=/tmp/recordHandoff.mjs && node src/lib/recordHandoff.test.mjs
import assert from 'node:assert/strict'

const { trainHandoff } = await import('/tmp/recordHandoff.mjs')

// the happy path: a finished dataset offers training, seeded with its real path
{
  const h = trainHandoff({ ok: true, dataset: 'cagatay/pick-cube', root: '/data/lerobot/cagatay/pick-cube', episodes_kept: 5 })
  assert.ok(h)
  assert.equal(h.prefill.dataset_root, '/data/lerobot/cagatay/pick-cube', 'the resolved path wins over the id')
  assert.match(h.label, /5 episodes/)
  assert.equal(h.caveat, null)
}

// no path from the server: the id still works as a seed (the trainer resolves it the same way)
{
  const h = trainHandoff({ ok: true, dataset: 'cagatay/pick-cube', episodes_kept: 1 })
  assert.equal(h.prefill.dataset_root, 'cagatay/pick-cube')
  assert.match(h.label, /1 episode\b/, 'singular for one')
}

// a dataset with zero kept episodes trains nothing — no offer, ever
assert.equal(trainHandoff({ ok: true, dataset: 'x/y', root: '/d', episodes_kept: 0 }), null)

// a failed close is not a dataset — no offer
assert.equal(trainHandoff({ ok: false, detail: 'finalize failed' }), null)
assert.equal(trainHandoff(null), null)

// the camera defect the dataset was born with RIDES WITH the offer, not hidden behind it
{
  const h = trainHandoff({
    ok: true, dataset: 'x/y', root: '/d', episodes_kept: 3,
    camera_notice: { present: [], missing: ['top', 'wrist'] },
  })
  assert.ok(h)
  assert.match(h.caveat, /cannot train a visual policy/)
}
{
  const h = trainHandoff({
    ok: true, dataset: 'x/y', root: '/d', episodes_kept: 3,
    camera_notice: { present: ['top'], missing: ['wrist'] },
  })
  assert.match(h.caveat, /missing/)
}

console.log('recordHandoff: all assertions passed')
