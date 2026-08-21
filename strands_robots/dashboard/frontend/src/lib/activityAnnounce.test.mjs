import assert from 'node:assert/strict'
const { activityAnnouncement } = await import('/tmp/activityAnnounce.mjs')

const OPENED = 1000

// --- nothing that predates the sheet is news -----------------------------------------
{
  assert.equal(activityAnnouncement(undefined, OPENED), '')
  assert.equal(activityAnnouncement(null, OPENED), '')
  const old = { t: OPENED - 5, source: 'api', action: 'stop', target: 'arm-1', ok: true }
  assert.equal(activityAnnouncement(old, OPENED), '',
    'the sheet loads history on open — reading it aloud would be a paragraph of stale speech')
  assert.equal(activityAnnouncement({ ...old, t: OPENED }, OPENED), '', 'the boundary is exclusive')
}

// --- a fresh line is spoken, with who did it and what the robot answered -------------
{
  const said = activityAnnouncement({ t: OPENED + 1, source: 'agent', action: 'stop', target: 'arm-1', ok: true }, OPENED)
  assert.match(said, /^agent stop on arm-1: /)
  assert.doesNotMatch(said, /failed|warning/, 'a success is not announced as trouble')
}

// --- THE POINT: a failure sounds like one ---------------------------------------------
{
  const bad = activityAnnouncement({ t: OPENED + 1, source: 'api', action: 'stop', target: 'arm-1', ok: false }, OPENED)
  assert.match(bad, /^failed — api stop on arm-1: the call failed$/,
    '"stop on arm-1" spoken flatly sounds like it worked')
}

// --- a fleet-wide action has no single target, and must not say "on —" ----------------
{
  const estop = activityAnnouncement(
    { t: OPENED + 2, source: 'estop', action: 'estop', target: '', ok: true, detail: { responses_received: 4 } }, OPENED)
  assert.doesNotMatch(estop, / on —/, 'an em dash target is a placeholder, not a robot name')
  assert.match(estop, /all peers/, 'the fleet-wide target is named in words')
  // An unproven stop is a warning, not a success: nobody acked it.
  const unproven = activityAnnouncement(
    { t: OPENED + 3, source: 'estop', action: 'estop', target: '', ok: true, detail: { responses_received: 0 } }, OPENED)
  assert.match(unproven, /^warning — /)
}

// --- whitespace never leaks into speech ----------------------------------------------
{
  const s = activityAnnouncement({ t: OPENED + 1, source: 'mesh', action: 'spawn', target: ' arm-2 ', ok: true }, OPENED)
  assert.doesNotMatch(s, /  |\s$/, 'double spaces are audible as a stumble')
}
console.log('activityAnnounce: ok')
