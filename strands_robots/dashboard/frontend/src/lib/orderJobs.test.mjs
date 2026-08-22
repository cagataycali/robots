// Assertions for the Jobs list order (lib/orderJobs.ts).
// Run: npx esbuild src/lib/orderJobs.ts --bundle --format=esm --outfile=/tmp/orderJobs.mjs \
//        && node src/lib/orderJobs.test.mjs
import assert from 'node:assert/strict'

const { orderJobsNewestFirst } = await import('/tmp/orderJobs.mjs')

const ids = (rows) => rows.map(r => r.job_id)

// The ordinary ledger: append order, newest last -> newest first out.
{
  const out = orderJobsNewestFirst([
    { job_id: 'old', submitted_at: 100 },
    { job_id: 'mid', submitted_at: 200 },
    { job_id: 'new', submitted_at: 300 },
  ])
  assert.deepEqual(ids(out), ['new', 'mid', 'old'])
}

// THE BUG THIS FILE EXISTS FOR: a ledger whose file order is NOT submission order
// (hand-edited, merged from two machines, restored from a quarantined copy).
{
  const out = orderJobsNewestFirst([
    { job_id: 'just-started', submitted_at: 900 },
    { job_id: 'ancient', submitted_at: 100 },
    { job_id: 'yesterday', submitted_at: 500 },
  ])
  assert.deepEqual(ids(out), ['just-started', 'yesterday', 'ancient'])
  // the poller takes the first five; the run the operator is watching must be in them
  assert.equal(out.slice(0, 5)[0].job_id, 'just-started')
}

// A legacy ledger with no timestamps anywhere behaves EXACTLY as the old reverse().
{
  const rows = [{ job_id: 'a' }, { job_id: 'b' }, { job_id: 'c' }]
  assert.deepEqual(ids(orderJobsNewestFirst(rows)), ['c', 'b', 'a'])
}

// Mixed: a row that can prove when it started outranks one that cannot, whatever the
// file order says.
{
  const out = orderJobsNewestFirst([
    { job_id: 'timed-old', submitted_at: 10 },
    { job_id: 'legacy-1' },
    { job_id: 'legacy-2' },
  ])
  assert.deepEqual(ids(out), ['timed-old', 'legacy-2', 'legacy-1'])
}

// Junk timestamps are not timestamps: 0, NaN, a string and null must not sort.
{
  const out = orderJobsNewestFirst([
    { job_id: 'zero', submitted_at: 0 },
    { job_id: 'nan', submitted_at: NaN },
    { job_id: 'str', submitted_at: '1700000000' },
    { job_id: 'null', submitted_at: null },
    { job_id: 'real', submitted_at: 5 },
  ])
  assert.equal(out[0].job_id, 'real')
  assert.equal(out.length, 5, 'nothing may be dropped')
}

// Same-second submissions keep the ledger's own order (stable sort), so the list does
// not shuffle on every refresh.
{
  const out = orderJobsNewestFirst([
    { job_id: 'first', submitted_at: 42 },
    { job_id: 'second', submitted_at: 42 },
  ])
  assert.deepEqual(ids(out), ['first', 'second'])
}

// Degenerate input is not a crash: the list is chrome around a poller.
{
  assert.deepEqual(orderJobsNewestFirst([]), [])
  assert.equal(orderJobsNewestFirst([null, { job_id: 'x', submitted_at: 3 }]).length, 2)
}

console.log('orderJobs: all assertions passed')
