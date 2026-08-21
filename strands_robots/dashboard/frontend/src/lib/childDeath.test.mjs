import assert from 'node:assert/strict'
import { deathVerdict, retainedOutputIsStartup } from '/tmp/childDeath.mjs'

// --- THE DEFECT: every death read as "exited" -----------------------------------
// The live case that cost two days: the twin's row said "· exited" for a kill -9.
{
  const v = deathVerdict(-9)
  assert.match(v.phrase, /killed \(SIGKILL\)/)
  assert.equal(v.unexplained, true, 'a SIGKILL does not say who sent it')
  assert.match(v.phrase, /nothing here sends that/, 'clear the dashboard, or the operator suspects the wrong thing')
  assert.doesNotMatch(v.phrase, /^exited/, 'the old word is what hid this for a day')
}

// A clean exit and a crash must not share a sentence with the kill -----------------
{
  const clean = deathVerdict(0)
  assert.match(clean.phrase, /cleanly/)
  assert.equal(clean.unexplained, false)
  const crash = deathVerdict(1)
  assert.match(crash.phrase, /code 1/, 'the number is the whole clue for a Python failure')
  assert.equal(crash.unexplained, false)
  assert.notEqual(clean.phrase, crash.phrase)
  assert.notEqual(crash.phrase, deathVerdict(-9).phrase)
}

// Named signals, and honesty about unnamed ones ------------------------------------
{
  assert.match(deathVerdict(-15).phrase, /SIGTERM/)
  assert.equal(deathVerdict(-15).unexplained, false, 'SIGTERM is what despawn sends — explainable')
  assert.match(deathVerdict(-11).phrase, /segfault/)
  assert.match(deathVerdict(-31).phrase, /signal 31/, 'report the number rather than invent a name')
  assert.equal(deathVerdict(-31).unexplained, true)
}

// "No status recorded" is NOT a clean exit ----------------------------------------
{
  for (const missing of [null, undefined]) {
    const v = deathVerdict(missing)
    assert.equal(v.unexplained, true)
    assert.doesNotMatch(v.phrase, /cleanly|code 0/, 'absence of evidence is not a tidy shutdown')
    assert.match(v.phrase, /no exit status/)
  }
}

// --- the second half of the lie: a startup burst posing as last words ------------
// Real shape: started 14:00:32 local, all 10 retained lines stamped 14:00:32.
{
  const startedAt = Date.parse('2026-08-20T18:00:32Z') / 1000
  const d = new Date(startedAt * 1000)
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  const lines = [`${hh}:${mm}:${ss} [safety:twin] No emergency-stop resume code set.`,
                 `${hh}:${mm}:${ss}   To allow remote resume: set STRANDS_MESH_OVERRIDE_CODE`]
  assert.equal(retainedOutputIsStartup({ lines, startedAt }), true)
}

// A child still talking hours later is NOT startup output -------------------------
{
  const startedAt = Date.parse('2026-08-20T18:00:32Z') / 1000
  const d = new Date((startedAt + 3 * 3600 + 47 * 60) * 1000)
  const late = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:00 sync read failed`
  assert.equal(retainedOutputIsStartup({ lines: ['00:00:00 boot', late], startedAt }), false,
    'one clocked line outside the window is enough — a spread ring means real output')
}

// No evidence => no claim (the caller renders nothing) ----------------------------
{
  assert.equal(retainedOutputIsStartup({ lines: [], startedAt: 1 }), null)
  assert.equal(retainedOutputIsStartup({ lines: ['no clock on this line'], startedAt: 1 }), null)
  assert.equal(retainedOutputIsStartup({ lines: ['00:00:01 x'], startedAt: null }), null)
  assert.equal(retainedOutputIsStartup({ lines: ['00:00:01 x'] }), null)
}

// The hour boundary must not fake a mismatch --------------------------------------
{
  const startedAt = Date.parse('2026-08-20T18:59:59Z') / 1000
  const d = new Date((startedAt + 2) * 1000)
  const line = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')} up`
  assert.equal(retainedOutputIsStartup({ lines: [line], startedAt }), true, '59:59 and 00:01 are 2s apart')
}
