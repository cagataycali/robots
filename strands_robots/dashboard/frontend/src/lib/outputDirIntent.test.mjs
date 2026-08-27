import assert from 'node:assert/strict'

const { outputDirSay, trainGate } = await import('/tmp/outputDirIntent.mjs')

// A free path is the normal case: say nothing, block nothing.
{
  const s = outputDirSay({ state: 'free', detail: 'does not exist yet — the run creates it' })
  assert.equal(s.text, null)
  assert.equal(s.confirmable, false)
  assert.equal(s.blocked, false)
}

// An occupied directory: the loudest tone, a tick, and a label that NAMES the loss.
{
  const s = outputDirSay({
    state: 'occupied', destructive: true, needs_confirm: true, path: '/Users/x/notes',
    total: 12, entries: ['a.md'], detail: 'holds 12 item(s) and NO training checkpoint, so starting a run here DELETES the directory',
  })
  assert.equal(s.tone, 'bad')
  assert.equal(s.confirmable, true)
  assert.equal(s.blocked, false)
  assert.match(s.confirmLabel, /delete 12 item\(s\) in \/Users\/x\/notes/)
  // "are you sure?" is not a label; the consequence is.
  assert.doesNotMatch(s.confirmLabel, /sure/i)
  assert.match(s.text, /DELETES/)
}

// A resumable directory cannot be confirmed into working — lerobot refuses the run.
{
  const s = outputDirSay({ state: 'resumable', detail: 'already holds a training checkpoint … pick a new directory' })
  assert.equal(s.blocked, true)
  assert.equal(s.confirmable, false, 'a tick here would authorise a run that cannot start')
}

// A file is a typo. Also blocked, also unconfirmable.
{
  const s = outputDirSay({ state: 'not_a_dir', detail: 'that path is a FILE' })
  assert.equal(s.blocked, true)
  assert.equal(s.confirmable, false)
}

// Unreadable: nobody can consent to a loss they were not shown.
{
  const s = outputDirSay({ state: 'unknown', detail: 'cannot read that path (PermissionError)' })
  assert.equal(s.blocked, true)
  assert.equal(s.confirmable, false)
}

// ------------------------------------------------------------------ the gate

// No path: the oldest rule on this form.
assert.equal(trainGate({ path: '  ', verdict: null, armedFor: null }).ok, false)

// No verdict yet (still typing / read in flight): allowed, and NOT carrying consent.
// A pending filesystem read must not look like a broken form; the backend still refuses.
{
  const g = trainGate({ path: '/tmp/new', verdict: null, armedFor: null, pending: true })
  assert.equal(g.ok, true)
  assert.equal(g.confirmClear, false)
}

// Occupied and unticked: refused, with the reason pointing at the tick.
{
  const g = trainGate({
    path: '/tmp/run', armedFor: null,
    verdict: { state: 'occupied', path: '/tmp/run', total: 3, needs_confirm: true, detail: 'DELETES' },
  })
  assert.equal(g.ok, false)
  assert.match(g.why, /tick the box/)
}

// Ticked FOR THIS PATH: allowed, and the request carries confirm_clear.
{
  const g = trainGate({
    path: '/tmp/run', armedFor: '/tmp/run',
    verdict: { state: 'occupied', path: '/tmp/run', total: 3, needs_confirm: true, detail: 'DELETES' },
  })
  assert.equal(g.ok, true)
  assert.equal(g.confirmClear, true)
}

// THE ONE THAT MATTERS: a tick made for one directory must never authorise deleting another.
// The operator ticks for /tmp/run-a, then edits the field to /tmp/run-b (also occupied).
{
  const g = trainGate({
    path: '/tmp/run-b', armedFor: '/tmp/run-a',
    verdict: { state: 'occupied', path: '/tmp/run-b', total: 40, needs_confirm: true, detail: 'DELETES' },
  })
  assert.equal(g.ok, false, 'consent is per-path — editing the field revokes it')
  assert.equal(g.confirmClear, false)
}

// A blocked state stays blocked even if something armed the tick earlier.
{
  const g = trainGate({
    path: '/tmp/old-run', armedFor: '/tmp/old-run',
    verdict: { state: 'resumable', path: '/tmp/old-run', detail: 'already holds a training checkpoint' },
  })
  assert.equal(g.ok, false)
  assert.equal(g.confirmClear, false)
  assert.match(g.why, /checkpoint/)
}

// A free path never sends consent it was not asked for.
{
  const g = trainGate({
    path: '/tmp/fresh', armedFor: '/tmp/fresh',
    verdict: { state: 'free', path: '/tmp/fresh', detail: 'does not exist yet' },
  })
  assert.equal(g.ok, true)
  assert.equal(g.confirmClear, false, 'confirm_clear on a free dir would be consent to nothing')
}

console.log('outputDirIntent: all assertions passed')
