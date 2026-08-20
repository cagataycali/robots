#!/usr/bin/env node
/**
 * Is the bundle the operator (and every page audit) actually sees BUILT FROM the current source?
 *
 * The dashboard serves dist/ from disk, so a rebuilt bundle goes live without restarting the server —
 * which is why this loop can land UI work at all under the never-restart law. The flip side is the
 * quietest false green in the project: edit src, forget `npm run build`, and every audit-*.mjs then
 * opens the OLD page, passes, and reports "verified ON THE PAGE". The audits are the only thing
 * standing between a claim and the truth, so they must refuse to run against a stale build rather
 * than confirm yesterday's bundle.
 *
 * Two independent questions, because they fail differently:
 *   1. is dist older than the source that produces it?  (forgot to build)
 *   2. does the RUNNING server serve the same entry asset dist/index.html names?  (serving another
 *      copy, or a stale service worker in front of it — this dashboard is a PWA, and its worker has
 *      already made a fixture invisible to playwright once)
 *
 * Test files and this scripts/ directory are excluded deliberately: they cannot reach the bundle, and
 * a guard that demands a rebuild after editing a test would be trained away within a day.
 *
 * Run: node scripts/check-dist-fresh.mjs [--url http://127.0.0.1:8090]
 */
import fs from 'node:fs'
import path from 'node:path'

const HERE = path.resolve(new URL('.', import.meta.url).pathname)
const ROOT = path.resolve(HERE, '..')
const urlArg = process.argv.indexOf('--url')
const BASE = urlArg > -1 ? process.argv[urlArg + 1] : (process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090')

const walk = (dir) => !fs.existsSync(dir) ? [] : fs.readdirSync(dir, { withFileTypes: true })
  .flatMap(e => e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)])

/** Source that can actually change the bundle. */
const sources = [
  ...walk(path.join(ROOT, 'src')).filter(f => !/\.test\.[mc]?[jt]sx?$/.test(f)),
  ...['index.html', 'vite.config.ts', 'vite.config.js'].map(f => path.join(ROOT, f)).filter(fs.existsSync),
]
const built = walk(path.join(ROOT, 'dist'))

if (!built.length) {
  console.error('  dist/ does not exist — the page an operator would open has never been built.')
  console.error('  fix: npm run build')
  process.exit(1)
}

const newest = (files) => files.reduce((best, f) => {
  const m = fs.statSync(f).mtimeMs
  return m > best.m ? { m, f } : best
}, { m: 0, f: '' })

const s = newest(sources), d = newest(built)
const rel = (f) => path.relative(ROOT, f)
let failed = false

if (s.m > d.m) {
  const age = Math.round((s.m - d.m) / 1000)
  console.error(`  STALE BUILD: ${rel(s.f)} is ${age}s newer than the newest file in dist/ (${rel(d.f)}).`)
  console.error('  Every page audit would open the OLD bundle and pass — that is a green about yesterday.')
  console.error('  fix: npm run build')
  failed = true
} else {
  console.log(`  dist is current (newest source ${rel(s.f)} ≤ ${rel(d.f)})`)
}

// Question 2: what does the live server actually hand out?
const entryOf = (html) => (html.match(/<script[^>]+src="([^"]+)"/) ?? [])[1] ?? null
try {
  // A 5s ceiling because this now runs INSIDE restart_dashboard.sh, the one command the owner types at
  // his desk. node's fetch has no default timeout, so a wedged or half-started server would hang the
  // restart itself — a guard that can stall the recovery it is part of is worse than no guard.
  const res = await fetch(`${BASE}/`, { headers: { 'Cache-Control': 'no-cache' }, signal: AbortSignal.timeout(5000) })
  const servedEntry = entryOf(await res.text())
  const diskEntry = entryOf(fs.readFileSync(path.join(ROOT, 'dist/index.html'), 'utf8'))
  if (servedEntry && diskEntry && servedEntry !== diskEntry) {
    console.error(`  SERVED MISMATCH: the server hands out ${servedEntry} but dist/index.html names ${diskEntry}.`)
    console.error('  A page audit would then verify a bundle that is not the one on disk. Likely a stale')
    console.error('  service worker or a second dist being served; hard-reload and re-check before trusting a pass.')
    failed = true
  } else if (servedEntry) {
    console.log(`  the running server serves the entry dist names (${servedEntry})`)
  }
} catch (e) {
  // Not this script's job to insist the dashboard is up: run-audits already refuses in that case, with
  // the terminal-restart wording. Silence here keeps one voice per problem.
  console.log(`  (server not reachable at ${BASE} — freshness of dist checked on disk only)`)
}

process.exit(failed ? 1 : 0)
