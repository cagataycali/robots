import assert from 'node:assert/strict'
import { absentNotice, quietNotice, shortCause } from '/tmp/absentChildren.mjs'

// --- THE DEFECT: a killed robot vanished with no word anywhere ---------------------
{
  const n = absentNotice([{ peer_id: 'so101-twin', returncode: -9, mode: 'sim' }])
  assert.ok(n, 'a dead child the dashboard started must reach the fleet view')
  assert.match(n.headline, /^so101-twin is gone/, 'the NAME leads: the operator is hunting one robot')
  assert.match(n.headline, /killed \(SIGKILL\)/)
  assert.equal(n.count, 1)
  // the bar gets one line, the tooltip carries the reasoning
  assert.ok(!n.headline.includes('nothing here sends that'), 'the long clause belongs in the tooltip')
  assert.match(n.detail, /nothing here sends that/, 'and it must not be lost')
}

// --- silence must stay silent ------------------------------------------------------
{
  assert.equal(absentNotice(undefined), null, 'an older server sends no field — that is not "all present"')
  assert.equal(absentNotice([]), null)
  assert.equal(absentNotice(null), null)
  assert.equal(absentNotice([{ peer_id: '' }]), null, 'an unnamed entry cannot be reported')
}

// --- an expected ending is not news ------------------------------------------------
{
  assert.equal(
    absentNotice([{ peer_id: 'collect-job', returncode: 0, mode: 'collect' }]), null,
    'a job that finished and left the mesh must not nag the bar, or the bar gets ignored',
  )
  const n = absentNotice([
    { peer_id: 'collect-job', returncode: 0 },
    { peer_id: 'arm-1', returncode: 1 },
  ])
  assert.equal(n.count, 1, 'only the surprise is counted')
  assert.match(n.headline, /^arm-1 is gone/)
  assert.ok(!n.detail.includes('collect-job'), 'the clean exit stays in the drawer ledger')
}

// --- several deaths ----------------------------------------------------------------
{
  const n = absentNotice([
    { peer_id: 'arm-1', returncode: -9 },
    { peer_id: 'arm-2', returncode: 1 },
  ])
  assert.equal(n.headline, '2 robots you started are gone')
  assert.match(n.detail, /arm-1 — killed \(SIGKILL\)/)
  assert.match(n.detail, /arm-2 — exited with code 1/)
  assert.equal(n.detail.split('\n').length, 2, 'one line per robot, so the tooltip is readable')
}

// --- a child that never got a status ------------------------------------------------
{
  // process is None: the spawn produced nothing. Still worth saying — the operator
  // clicked start and no robot ever appeared — but it must not claim a cause.
  const n = absentNotice([{ peer_id: 'never-started', returncode: null }])
  assert.ok(n, 'a spawn that produced nothing is exactly what the operator is confused by')
  assert.doesNotMatch(n.headline, /killed|crashed|code/, 'no cause may be invented from a missing status')
}

// --- shortCause ---------------------------------------------------------------------
{
  assert.equal(shortCause(-9), 'killed (SIGKILL)')
  assert.equal(shortCause(0), shortCause(0).split(' — ')[0], 'idempotent on a clause-free phrase')
}
// --- quietNotice (Q155b): alive, ours, and absent -----------------------------------
{
  assert.equal(quietNotice(undefined), null, 'an older server that sends no field claims nothing')
  assert.equal(quietNotice([]), null, 'no quiet children is not news')
  assert.equal(quietNotice(['', null]), null, 'blank ids are not robots')

  const one = quietNotice(['sim-a'])
  assert.ok(one && one.count === 1)
  assert.match(one.headline, /^sim-a /, 'the name comes first — the operator is hunting one robot')
  assert.doesNotMatch(one.headline, /gone|died|crashed|killed/,
    'THE POINT: this process is ALIVE, and calling it gone sends the operator to the wrong remedy')
  assert.match(one.detail, /process is running/, 'the tooltip must say what is true of it')

  const many = quietNotice(['sim-a', 'sim-b'])
  assert.match(many.headline, /^2 robots/)
  assert.equal(many.detail.split('\n').length, 2, 'one tooltip line per robot')

  // Disjointness, defended on this side too: the server cannot put an id in both lists,
  // but if it ever did, one robot must not produce two chips.
  const both = quietNotice(['sim-a'], [{ peer_id: 'sim-a', returncode: -9 }])
  assert.equal(both, null, 'death is the more specific claim and wins')
  const mixed = quietNotice(['sim-a', 'sim-b'], [{ peer_id: 'sim-a', returncode: 1 }])
  assert.ok(mixed && mixed.count === 1 && mixed.headline.startsWith('sim-b'),
    'only the genuinely quiet one survives the overlap')
}

console.log('absentChildren: ok')
