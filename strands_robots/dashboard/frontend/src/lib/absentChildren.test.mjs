import assert from 'node:assert/strict'
import { absentNotice, shortCause } from '/tmp/absentChildren.mjs'

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
console.log('absentChildren: ok')
