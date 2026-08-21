/**
 * Run the page audits — the harness they never had either.
 *
 * There are 20 `audit-*.mjs` scripts in here, each proving on the REAL page a claim some commit made
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

let scripts = fs.readdirSync(HERE)
  .filter(f => /^audit-.*\.mjs$/.test(f))
  .filter(f => !filters.length || filters.some(x => f.includes(x)))
  .sort()

// A filter that matches nothing must not look like a clean sweep: "PASS 0 audits" is the most
// dangerous possible output for a verification tool.
if (filters.length && !scripts.length) {
  console.error(`  no audit matches ${filters.join(', ')} — nothing was verified`)
  process.exit(1)
}
if (filters.length) console.log(`  filter ${filters.join(', ')} — ${scripts.length} of 16 audit(s)`)

/** The one-line purpose the script states about itself (first prose line of its header). */
const purpose = (f) => {
  const head = fs.readFileSync(path.join(HERE, f), 'utf8').split('\n').slice(0, 8).join('\n')
  const line = head.split('\n').map(l => l.replace(/^\s*(\/\*+|\*+\/?)\s?/, '').trim())
    .find(l => l.length > 20 && !l.startsWith('Run:') && !/^import/.test(l))
  return (line ?? '').replace(/\s+$/, '').slice(0, 96)
}

let failed = 0, timedOut = 0
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
    console.log(`  ok    ${name} (${secs}s) — ${purpose(f)}`)
    // NEWS FROM A PASSING AUDIT. Not every audit produces a verdict: audit-server-age exists to
    // REPORT which routes the shipped bundle calls that the running server lacks, and an old server
    // is news for the operator, not a broken build — so it exits 0 by design. This runner printed
    // only "ok" and threw the finding away, which is how a sweep reported "ok server-age (0s)" while
    // NINE routes were dark on the live dashboard. A line starting NEWS/OLD/note now survives.
    for (const line of (r.stdout || '').split('\n').filter(l => /^\s*(NEWS|OLD|note)\b/.test(l)).slice(0, 12))
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
  : `\n  PASS  ${scripts.length} audits in ${mins} min — every page claim still holds`)
process.exit(failed ? 1 : 0)
