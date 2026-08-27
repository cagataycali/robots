// Assertions for the Jobs list's ledger notice (lib/jobsLedger.ts).
// Run: npx esbuild src/lib/jobsLedger.ts --bundle --format=esm --outfile=/tmp/jobsLedger.mjs \
//        && node src/lib/jobsLedger.test.mjs
import assert from 'node:assert/strict'

const { jobsLedgerNotice } = await import('/tmp/jobsLedger.mjs')

// Jobs present and the ledger is fine: the list needs no caption.
{
  const v = jobsLedgerNotice({ count: 3 })
  assert.equal(v.text, null)
  assert.equal(v.partial, false)
}

// An honestly empty history is an ordinary answer, not a warning.
{
  const v = jobsLedgerNotice({ count: 0, problem: null })
  assert.equal(v.tone, 'info')
  assert.equal(v.partial, false)
  assert.match(v.text, /No training jobs yet/)
}

// An unreadable ledger with NO rows must not read like "nothing ever ran".
{
  const v = jobsLedgerNotice({ count: 0, problem: 'the training job history could not be read (JSONDecodeError) and was moved to /tmp/x.corrupt-1 — runs started before now have no card here, but any that are still running are unaffected' })
  assert.equal(v.tone, 'warn')
  assert.equal(v.partial, true)
  assert.match(v.text, /not because nothing ran/)
  // the API's own sentence is quoted, not paraphrased: one wording, one truth
  assert.match(v.text, /still running are unaffected/)
  assert.match(v.text, /moved to \/tmp\/x\.corrupt-1/)
}

// The dangerous case: SOME rows render, so the list looks complete and the missing
// runs look like runs that never existed.
{
  const v = jobsLedgerNotice({ count: 2, problem: 'the training job history could not be read' })
  assert.equal(v.partial, true)
  assert.match(v.text, /Some earlier runs may be missing/)
}

// Whitespace is not a problem report.
{
  const v = jobsLedgerNotice({ count: 0, problem: '   ' })
  assert.equal(v.partial, false)
  assert.match(v.text, /No training jobs yet/)
}

console.log('jobsLedger: all assertions passed')
