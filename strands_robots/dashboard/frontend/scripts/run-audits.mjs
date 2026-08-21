/**
 * Run the page audits — the harness they never had either.
 *
 * Every `audit-*.mjs` in here is run — the count lives in the code below (readdirSync + a glob), and
 * this sentence deliberately no longer states it. It said 20, was corrected to 28, and was 34 by the
 * time anyone looked again: a number in prose beside a directory that grows weekly is a fact with a
 * half-life. Each audit proves on the REAL page a claim some commit made
 * (a dropped camera is not an absent one, an abandoned dataset directory is not a dataset, a training
 * provider that cannot run is not offered, the devices screen's three empty worlds differ). `npm run
 * audit` chained exactly four of them, so the other eleven ran once, in the iteration that wrote them,
 * and then only if someone remembered the filename. Same disease as the lib tests (9fdfa887): the
 * proof existed and was unreachable.
 *
 * Every audit was checked to be READ-ONLY before this runner was written: none issues a POST/PUT/DELETE
 * and none commands a robot — they navigate, and several replace their OWN browser's view of an endpoint
 * with page.route. That is why running them all is safe with cagatay away. Anything that ever needs to
 * mutate must not be discovered by this file.
 *
 * Each script's purpose is read from its own header comment, so this runner cannot drift from them.
 * A per-script timeout is enforced: an audit that hangs (a route fixture that never resolves, a browser
 * that never launches) would otherwise take the whole sweep with it and report nothing at all.
 *
 *   npm run audit:all                 # every audit
 *   node scripts/run-audits.mjs cam   # only those whose filename matches
 *   AUDIT_TIMEOUT_S=240 npm run audit:all
 */
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const HERE = new URL('.', import.meta.url).pathname
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const TIMEOUT_S = Number(process.env.AUDIT_TIMEOUT_S ?? 180)
// Substring filters: `run-audits server-age` re-runs one audit, and SEVERAL are OR'd
// (`run-audits record training`) so a change touching two screens can be re-verified in seconds
// instead of waiting out sixteen browser launches. One arg has always worked; iteration 217 added the
// rest after writing a second filter that silently shadowed this one and reported "1 of 1".
const filters = process.argv.slice(2).filter(a => !a.startsWith('-'))

// These need the live dashboard. Say so once, here, instead of 15 browser launches failing obscurely.
try {
  const r = await fetch(BASE, { signal: AbortSignal.timeout(4000) })
  if (!r.ok && r.status !== 401) throw new Error(`HTTP ${r.status}`)
} catch (e) {
  console.error(`  the dashboard at ${BASE} did not answer (${e.message ?? e}) — these audits read the`)
  console.error('  running page, so there is nothing to audit. Start it from a TERMINAL (camera TCC law)')
  console.error('  and re-run; never restart it from a background loop.')
  process.exit(2)
}

// PREFLIGHT: these audits read the SERVED page, so a dist older than src means every one of them
// verifies the previous bundle and passes — the most expensive false green available here, because the
// audits are the only thing standing between a claim and the truth. Refuse instead of confirming
// yesterday, with the same exit 2 the unreachable-dashboard refusal uses (a setup problem, not a
// failing claim). AUDIT_SKIP_FRESHNESS=1 exists for auditing a deliberately old build.
if (process.env.AUDIT_SKIP_FRESHNESS !== '1') {
  const fresh = spawnSync(process.execPath, [new URL('check-dist-fresh.mjs', import.meta.url).pathname,
                                             '--url', BASE], { encoding: 'utf8' })
  process.stdout.write(fresh.stdout || '')
  if (fresh.status !== 0) {
    process.stderr.write(fresh.stderr || '')
    console.error('  refusing to audit a bundle that is not built from the current source.')
    process.exit(2)
  }
}

const allAudits = fs.readdirSync(HERE).filter(f => /^audit-.*\.mjs$/.test(f)).sort()
let scripts = allAudits.filter(f => !filters.length || filters.some(x => f.includes(x)))

// A filter that matches nothing must not look like a clean sweep: "PASS 0 audits" is the most
// dangerous possible output for a verification tool.
if (filters.length && !scripts.length) {
  console.error(`  no audit matches ${filters.join(', ')} — nothing was verified`)
  process.exit(1)
}
// "X of Y" must be COUNTED, not remembered: this said "of 16" while the directory held 28. A
// narrowed run that misstates the whole is how a partial sweep gets read as a full one — the same
// law the python audit scripts follow after scripts/audit_collaborator_kwargs.py printed a
// confident total for 2 of the 7 classes it exists to check.
if (filters.length) console.log(`  filter ${filters.join(', ')} — ${scripts.length} of ${allAudits.length} audit(s)`)

/** The one-line purpose the script states about itself (first prose line of its header). */
const purpose = (f) => {
  const head = fs.readFileSync(path.join(HERE, f), 'utf8').split('\n').slice(0, 8).join('\n')
  const line = head.split('\n').map(l => l.replace(/^\s*(\/\*+|\*+\/?)\s?/, '').trim())
    .find(l => l.length > 20 && !l.startsWith('Run:') && !/^import/.test(l))
  return (line ?? '').replace(/\s+$/, '').slice(0, 96)
}

let failed = 0, timedOut = 0, narrowed = 0
const started = Date.now()
for (const f of scripts) {
  const t0 = Date.now()
  const r = spawnSync(process.execPath, [path.join(HERE, f)], {
    encoding: 'utf8', timeout: TIMEOUT_S * 1000, killSignal: 'SIGKILL',
  })
  const secs = ((Date.now() - t0) / 1000).toFixed(0)
  const name = f.replace(/^audit-|\.mjs$/g, '')
  if (r.error?.code === 'ETIMEDOUT' || r.signal === 'SIGKILL') {
    timedOut += 1; failed += 1
    console.log(`  HUNG  ${name} — killed after ${TIMEOUT_S}s (a hang is a failure, not a skip)`)
    continue
  }
  if (r.status === 0) {
    // A GREEN THAT PROVES LESS THAN IT USED TO. Measured 2026-08-22 on cagatay's own fleet: both real
    // arms had been silent for three days and the sim twin was gone, so audit-record-joint-warning
    // could exercise NEITHER its warning case nor its control case — and still printed a plain `ok`.
    // The suite is where this repo's "a tool that can be NARROWED must say X of Y" law was written,
    // and it was the one place not obeying it: a degraded fleet made the sweep QUIETER instead of
    // louder, and the day the hardware returns is the day those cases silently resume with nobody
    // told they had stopped. An audit declares that with a NARROWED: line — a PREFIX it opts into,
    // never prose this runner pattern-matches, because guessing at English is how a bypass grows.
    // Deliberately NOT a failure: a fleet with no arms is a legitimate state (Q135 first-run), and a
    // suite that goes red when the hardware is off trains the next agent to ignore it.
    // SKIP IS NARROWING, SPELLED DIFFERENTLY — and it was invisible to this summary until 2026-08-22.
    // Four audits already declared a skipped case with a `SKIP` prefix and the sweep printed a plain
    // `ok` over all of them: audit-server-age SKIPs when it cannot reach the server AT ALL (a green
    // that measured nothing, in the audit whose entire purpose is comparing bundle to server),
    // audit-touch-targets SKIPs a WHOLE SCREEN when its nav chip is absent, audit-screens-render
    // SKIPs a screen "not on this rig", audit-calibrate-command skips its phone pass with no
    // calibrate row. Counting them as narrowing costs one regex and turns four silent holes into
    // sentences; no audit had to be rewritten, because the prefix was already there.
    const narrowedLines = (r.stdout || '').split('\n').filter(l => /^\s*(NARROWED|SKIP)\b/.test(l))
    if (narrowedLines.length) narrowed += 1
    console.log(`  ok${narrowedLines.length ? '~' : '  '}  ${name} (${secs}s) — ${purpose(f)}`)
    // NEWS FROM A PASSING AUDIT. Not every audit produces a verdict: audit-server-age exists to
    // REPORT which routes the shipped bundle calls that the running server lacks, and an old server
    // is news for the operator, not a broken build — so it exits 0 by design. This runner printed
    // only "ok" and threw the finding away, which is how a sweep reported "ok server-age (0s)" while
    // NINE routes were dark on the live dashboard. A line starting NEWS/OLD/note now survives.
    for (const line of (r.stdout || '').split('\n').filter(l => /^\s*(NEWS|OLD|note|NARROWED|SKIP)\b/.test(l)).slice(0, 12))
      console.log(`          ${line.trim()}`)
  } else {
    failed += 1
    console.log(`  FAIL  ${name} (${secs}s) — ${purpose(f)}`)
    for (const line of (r.stdout || '').split('\n').filter(l => /FAIL|✗|Error/.test(l)).slice(0, 5))
      console.log(`          ${line.trim()}`)
    for (const line of (r.stderr || '').trim().split('\n').filter(Boolean).slice(0, 3))
      console.log(`          ${line.trim()}`)
  }
}

const mins = ((Date.now() - started) / 60000).toFixed(1)
console.log(failed
  ? `\n  ${failed} of ${scripts.length} audit(s) FAILED in ${mins} min${timedOut ? ` (${timedOut} hung)` : ''}`
  : narrowed
    ? `\n  PASS  ${scripts.length} audits in ${mins} min — but ${narrowed} was NARROWED (marked ok~):\n        part of its case could not be exercised on this fleet, so this green covers less than\n        it did. The remedy is the fleet, not the audit.`.replace(' was NARROWED', narrowed === 1 ? ' was NARROWED' : ' were NARROWED').replace('part of its case', narrowed === 1 ? 'part of its case' : 'part of their case')
    : `\n  PASS  ${scripts.length} audits in ${mins} min — every page claim still holds`)
process.exit(failed ? 1 : 0)
