import assert from 'node:assert/strict'

const { labelsGate, labelSummary, labelRowLine } = await import('/tmp/episodeLabels.mjs')

// A Hub row has no local sidecar. Offering the button would fetch a 404 and blame the dataset.
{
  const g = labelsGate({ root: null })
  assert.equal(g.ok, false)
  assert.match(g.reason, /sidecar next to the dataset on disk/)
  assert.match(g.reason, /download this Hub dataset first/)
}

// Mid-recording is not a refusal about permissions: nothing has been judged yet.
{
  const g = labelsGate({ root: '/ds/x', recording: true })
  assert.equal(g.ok, false)
  assert.match(g.reason, /judged after the session/)
}

{
  assert.equal(labelsGate({ root: '/ds/x' }).ok, true)
  assert.equal(labelsGate(null).ok, false)
}

// THE RULE THIS FILE EXISTS FOR: when labelling is impossible, show the server's sentence, not a
// zero. "0 labelled" and "this kind of recording cannot be labelled" mean opposite things.
{
  const s = labelSummary({
    episodes: [], with_verdict: 0, labelled: 0, disputed: 0, can_annotate: false,
    why: 'a real-arm recording has no predicate verdict to annotate — this is a gap in the label rail, not a permission problem',
  })
  assert.match(s.text, /not a permission problem/)
  assert.ok(!/^0\//.test(s.text), 'must not reduce an impossibility to a count')
}

// A corrupt sidecar is a WARNING and a different action from "record verdicts first".
{
  const s = labelSummary({
    episodes: [], with_verdict: 0, labelled: 0, disputed: 0, can_annotate: false,
    sidecar_error: 'JSONDecodeError: line 3', why: 'could not be read (JSONDecodeError: line 3)',
  })
  assert.equal(s.tone, 'warn')
  assert.match(s.text, /may be damaged/)
}

// Counts appear only when labelling is actually possible, and a dispute raises the tone.
{
  const view = {
    benchmark: 'cube_lift', episodes: [{ annotatable: true, quality: 'high' }],
    with_verdict: 3, labelled: 1, disputed: 1, total_episodes: 5, can_annotate: true, why: 'x',
  }
  const s = labelSummary(view)
  assert.match(s.text, /1\/3 judged/)
  assert.match(s.text, /5 episodes recorded/)
  assert.match(s.text, /benchmark cube_lift/)
  assert.match(s.text, /1 disputing the verdict/)
  assert.equal(s.tone, 'warn')
}

// A transport failure is not silence, and a pending read is not "no labels".
{
  assert.equal(labelSummary(null, 'network down').tone, 'warn')
  assert.match(labelSummary(null).text, /Reading labels/)
}

// An unjudged-but-verdicted episode says what is missing; an unverdicted one says why it is stuck.
{
  const waiting = labelRowLine({ episode_index: 1, verdict: 'failure', quality: null, failure_mode: null, note: null, disputes_verdict: false, model: null, annotatable: true })
  assert.equal(waiting.badge, '✗')
  assert.match(waiting.detail, /awaiting a quality grade/)
  assert.equal(waiting.muted, true)

  const stuck = labelRowLine({ episode_index: 2, verdict: null, quality: null, failure_mode: null, note: null, disputes_verdict: false, model: null, annotatable: false })
  assert.match(stuck.detail, /cannot be annotated/)
  assert.equal(stuck.badge, '—')

  const judged = labelRowLine({ episode_index: 0, verdict: 'success', quality: 'high', failure_mode: 'near_miss', note: 'clean', disputes_verdict: true, model: 'human', annotatable: true })
  assert.equal(judged.badge, '✓')
  assert.match(judged.detail, /quality high/)
  assert.match(judged.detail, /near_miss/)
  assert.match(judged.detail, /judge disputes this verdict/)
  assert.match(judged.detail, /by human/)
  assert.equal(judged.muted, false)
}

console.log('episodeLabels: all assertions passed')
