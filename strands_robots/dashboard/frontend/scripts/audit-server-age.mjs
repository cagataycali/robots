#!/usr/bin/env node
/**
 * Which routes does the CURRENT SOURCE register that the RUNNING server does not have? (Q79)
 *
 * The dashboard process lives for days while this bundle is rebuilt on every landed change, so the two
 * halves are routinely different ages and the symptom is a bare 404 that blames the resource. This reads
 * the running server's own /openapi.json and diffs it against the routes the current source registers —
 * no POSTing, no side effects, safe against a live fleet (never probe /api/safety/estop to see if it is
 * there).
 *
 * Run: node scripts/audit-server-age.mjs        (env: STRANDS_DASH_URL, STRANDS_DASH_TOKEN_FILE)
 * Exit 0 always: an old running server is news for the operator, not a broken build.
 */
import fs from 'node:fs'
import path from 'node:path'

const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const TOKEN_FILE = process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`
const token = fs.existsSync(TOKEN_FILE) ? fs.readFileSync(TOKEN_FILE, 'utf8').trim() : ''

// EVERY python file of the dashboard package, not just server.py: /api/record/* is registered in
// record_api.py on its own APIRouter, and scanning one file made those eight routes look like things
// the RUNNING SERVER had invented — with a note politely explaining them as "registered elsewhere".
// They were registered elsewhere in the SOURCE, and the frontend calls all eight (src/lib/recordApi.ts).
// The old shape also made this audit's headline claim false: reading server.py alone cannot tell you
// what the BUNDLE calls, only what one module declares. Two lessons of this week at once — a report
// that misdescribes its own method, and a sweep whose "extra" bucket was an artifact of its input.
const pkgDir = path.resolve('../')
const pyFiles = fs.readdirSync(pkgDir).filter(f => f.endsWith('.py')).map(f => path.join(pkgDir, f))
const ambiguous = []
const declared = [...new Set(pyFiles.flatMap(f => {
  const body = fs.readFileSync(f, 'utf8')
  // A route path is only absolute when the decorated object has no prefix. record_api.py builds
  // APIRouter(prefix="/api/record") and decorates @r.post("/episode/redo"), so a scan that demands
  // "/api" in the decorator finds NONE of its nine routes and then reports them as things the server
  // invented. Resolve the prefix instead.
  const prefixes = [...new Set([...body.matchAll(/APIRouter\(\s*prefix\s*=\s*["']([^"']+)["']/g)].map(m => m[1]))]
  const paths = [...body.matchAll(/@\w+\.(?:get|post|put|delete|patch)\(\s*["']([^"']*)["']/g)].map(m => m[1])
  return paths.flatMap(raw => {
    if (raw.startsWith('/api')) return [raw]
    if (prefixes.length === 1) return [`${prefixes[0]}${raw}`.replace(/\/$/, '')]
    if (prefixes.length > 1) { ambiguous.push(`${path.basename(f)}${raw}`); return [] }
    return []  // a non-/api route (static, health) — not this audit's subject
  })
}))]

let live
try {
  const res = await fetch(`${BASE}/openapi.json`, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
  if (!res.ok) throw new Error(`HTTP ${res.status} — is the token current?`)
  live = Object.keys((await res.json()).paths ?? {})
} catch (e) {
  console.log(`  SKIP  no server to compare at ${BASE}: ${e.message}`)
  process.exit(0)
}

const missing = declared.filter(p => !live.includes(p)).sort()
const stale = live.filter(p => p.startsWith('/api/') && !declared.includes(p)).sort()

console.log(`  server at ${BASE} publishes ${live.length} paths; this source registers ${declared.length}`
            + ` across ${pyFiles.length} python modules`)
if (missing.length === 0) {
  console.log('  PASS  the running server has every route this source calls')
  // Same disclosure on the GREEN path, where it matters more: "every route" is not "every feature".
  console.log('        (paths only — a new FIELD on an existing route is outside this audit\'s reach)')
} else {
  // Prefixed NEWS so a full sweep surfaces it: this audit's whole output IS its value, and exiting
  // 0 previously made the runner swallow every line of it.
  console.log(`  NEWS  ${missing.length} route(s) this bundle calls are NOT on the running server — restart to light them up:`)
  // Every line names its OWN direction. The two lists used to print as bare paths, which made the
  // first list's items sit directly above the SECOND list's summary — and a tail read (how a loop
  // consumes a 4.5-minute sweep log) then attributed them to the opposite direction. It fooled the
  // person who wrote this sweep, one iteration after writing it, so the report is at fault: a line
  // that only means the right thing when you can see what is above it does not survive truncation.
  for (const p of missing) console.log(`  note    bundle needs → ${p}`)
  console.log('        → the UI explains these as "restart the dashboard to pick it up" (lib/serverAge.ts).')
  console.log('        → an owner-run restart from a terminal makes them live (never restart from a daemon:')
  console.log('          a launchd-descended process can never be granted camera access on macOS).')
  // WHAT THIS AUDIT CANNOT SEE, said out loud rather than left to be assumed: it compares PATHS.
  // A route that already exists but has grown a new RESPONSE FIELD (or accepts a new request field)
  // is invisible here — /api/config gaining `security.notice` in 40b68667 is exactly that shape, and
  // a reader of a green "every route present" line would conclude the server is current when a whole
  // feature is still dark. The rule in this repo: a check that can be narrower than its headline must
  // say so where the headline is printed.
  console.log('        → SCOPE: paths only. A new FIELD on an existing route cannot be detected here;')
  console.log('          RESTART_NOTES.md carries the field-level list (e.g. /api/config security.notice).')
}
// Honesty about the scan's own limits: a module with SEVERAL routers cannot be attributed by regex, and
// silently dropping those routes would turn them into phantom "server only" entries — the exact bug this
// commit fixes. Name them instead so the next reader knows the input was narrowed.
if (ambiguous.length) console.log(`  note  ${ambiguous.length} route(s) skipped: more than one APIRouter prefix in the file (${ambiguous.slice(0, 3).join(', ')}…)`)
if (stale.length) {
  console.log(`  note  ${stale.length} route(s) the RUNNING SERVER has that this source does not declare`)
  console.log('        (registered by another router, or removed from the source since that server started):')
  for (const p of stale) console.log(`  note    server only  → ${p}`)
}
