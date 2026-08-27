import assert from 'node:assert/strict'
const { jobTransitions, isTerminal, shortJob } = await import('/tmp/jobAnnounce.mjs')

// --- THE DEFECT: the end of an hours-long run had no signal --------------------------
{
  assert.equal(jobTransitions({ a: 'running' }, { a: 'succeeded' }), 'training job finished: a succeeded')
  assert.match(jobTransitions({ a: 'running' }, { a: 'failed' }), /^training job failed: a failed$/)
}

// --- a polled state must not repeat itself forever ----------------------------------
{
  assert.equal(jobTransitions({ a: 'running' }, { a: 'running' }), '', 'no change, no speech')
  assert.equal(jobTransitions({ a: 'succeeded' }, { a: 'succeeded' }), '',
    'already announced — a live region would say it on every poll')
}

// --- a job already finished when the tab opened is not news --------------------------
{
  assert.equal(jobTransitions({}, { a: 'failed' }), '',
    'first sight of a terminal job means it ended before we looked (the Q158 rule)')
  // …but once seen running, its ending IS news.
  assert.match(jobTransitions({ a: 'running' }, { a: 'error' }), /a error/)
}

// --- several at once: ONE sentence, failures first -----------------------------------
{
  const both = jobTransitions({ a: 'running', b: 'running' }, { a: 'succeeded', b: 'failed' })
  assert.match(both, /^training jobs ended, some badly: b failed, a succeeded$/,
    'the half that survives a cut-off should be the half needing a human')
  const two_ok = jobTransitions({ a: 'running', b: 'running' }, { a: 'succeeded', b: 'finished' })
  assert.match(two_ok, /^training jobs finished: a succeeded, b finished$/)
}

// --- vocabulary and ids --------------------------------------------------------------
{
  for (const s of ['succeeded', 'success', 'completed', 'finished', 'failed', 'error', 'cancelled', 'stopped', 'killed'])
    assert.equal(isTerminal(s), true, `${s} is terminal`)
  for (const s of ['running', 'queued', 'starting', '…', '', null, undefined])
    assert.equal(isTerminal(s), false, `${s} is in flight`)
  assert.equal(isTerminal('FAILED'), true, 'providers are not consistent about case')
  assert.equal(shortJob('abc'), 'abc')
  assert.equal(shortJob('0123456789abcdef-uuid'), '01234567…', 'a uuid read aloud in full is unusable')
}
console.log('jobAnnounce: ok')
