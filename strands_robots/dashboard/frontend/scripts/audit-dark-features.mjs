/**
 * The dark-feature banner tells the truth about THIS server — proved in a browser (Q124).
 *
 * lib/darkFeatures has unit assertions for the rule; what only a browser can answer is whether the
 * verdict reaches the DOM, and whether it reaches it CORRECTLY. Both halves matter and one of them
 * has already failed: the first live run showed ELEVEN dark features, two of which were invented by
 * treating '/api/auth/login/' (a base callers append 'begin' to) as a route. A banner that accuses a
 * healthy server of a missing feature is worse than no banner at all, and no unit test caught it —
 * the fixture did not contain a base until the browser said so.
 *
 * The expected set is computed with THE PRODUCT'S OWN RULE (darkFeatures.ts, esbuilt) against the
 * running server's real openapi.json. Re-implementing the comparison here would make this audit a
 * second source of truth that can agree with a bug.
 *
 * WHAT THIS THEREFORE CANNOT CATCH, stated so nobody trusts it further than it goes: a wrong RULE
 * that the page renders faithfully. Both sides read the same module, so they would agree. The rule
 * is covered by src/lib/darkFeatures.test.mjs (including the base case the browser taught us). What
 * this catches is the SCREEN and the RULE disagreeing — a shipped bundle older than the rule, a
 * banner that renders nothing, a count that does not match its own list, and above all a list with
 * an EXTRA route in it. Mutation-verified: reintroducing the base bug in the source made the running
 * page and the freshly-built rule disagree, and this audit exited 1.
 *
 * NARROWED, not passed, when the server has every route the bundle calls: then the only observable
 * claim is the ABSENCE of the banner, which is worth checking (a banner that appears on a current
 * server is a false alarm) but leaves the interesting case unexercised. That is the fleet's state,
 * not the audit's fault — restarting the dashboard is exactly what makes it narrow.
 *
 * Read-only: it opens one page and reads text. Nothing is spawned, nothing moves.
 * Run: node scripts/audit-dark-features.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations
import fs from 'node:fs'
import { execFileSync } from 'node:child_process'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

// The product's rule and the product's list, built the way the lib tests build them.
execFileSync('npx', ['esbuild', 'src/lib/darkFeatures.ts', '--bundle', '--format=esm',
  '--outfile=/tmp/audit-darkFeatures.mjs', '--log-level=error'], { stdio: 'inherit' })
const { darkRoutes } = await import('/tmp/audit-darkFeatures.mjs')

let live
try {
  const res = await fetch(`${BASE}/openapi.json`, { headers: { Authorization: `Bearer ${TOKEN}` } })
  if (!res.ok) throw new Error(`HTTP ${res.status} — is the token current?`)
  live = Object.keys((await res.json()).paths ?? {})
} catch (e) {
  console.log(`  SKIP  no server to compare at ${BASE}: ${e.message}`)
  process.exit(0)
}
const expected = darkRoutes(live)
console.log(`  note  server publishes ${live.length} paths; the bundle's rule calls ${expected.length} of its own routes dark`)

const browser = await chromium.launch()
// serviceWorkers must be BLOCKED: this is a PWA, and a service-worker-served response is not
// interceptable — a page-level audit that forgets this measures a cached yesterday.
const ctx = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1280, height: 1000 } })
const page = await ctx.newPage()
await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
// The banner waits on openapi.json, a request the page makes itself. Reading before it lands measures
// a race, and a race in an AUDIT invents a defect in the thing being audited (the role-badge lesson).
await page.waitForResponse(r => r.url().includes('/openapi.json'), { timeout: 20_000 })
  .catch(() => console.log('  note  the page never fetched openapi.json — it cannot know, so it must stay silent'))
await page.waitForTimeout(1500)

const banner = page.locator('div.toast.warn', { hasText: 'Older server' }).first()
const shown = await banner.count() > 0

if (expected.length === 0) {
  if (shown) failures.push(`the server has every route this bundle calls, yet the page claims: ${await banner.innerText()}`)
  else console.log('  note  no banner, and nothing to warn about — the two agree')
  console.log('  NARROWED: this server serves every route the bundle calls, so the banner\'s CONTENT could not be checked')
} else if (!shown) {
  failures.push(`${expected.length} route(s) are missing from this server (${expected.slice(0, 3).join(', ')}…) and the page says NOTHING — the operator meets each one as a 404 mid-task`)
} else {
  const text = (await banner.innerText()).replace(/\s+/g, ' ')
  console.log(`  note  banner: ${text.slice(0, 150)}`)
  if (!text.includes(`${expected.length} feature`)) failures.push(`banner does not name the real count (${expected.length}): ${text.slice(0, 120)}`)
  if (!/terminal/.test(text)) failures.push('the banner states a problem without its remedy (restart from a terminal)')
  // The list is the part an operator quotes into a bug report, so it must be exact — not a subset,
  // not a superset. A superset is the false accusation this audit exists to catch.
  await page.locator('summary', { hasText: /which one/ }).first().click().catch(() => {})
  await page.waitForTimeout(300)
  const listed = (await banner.innerText()).split('\n').map(l => l.trim()).filter(l => l.startsWith('/api'))
  const extra = listed.filter(p => !expected.includes(p))
  const absent = expected.filter(p => !listed.includes(p))
  if (extra.length) failures.push(`the banner names ${extra.length} route(s) the server DOES have: ${extra.join(', ')}`)
  if (absent.length) failures.push(`${absent.length} dark route(s) are missing from the list: ${absent.join(', ')}`)
  if (!extra.length && !absent.length) console.log(`  note  the list is exactly the ${listed.length} dark route(s): ${listed.join(' ')}`)
}

await browser.close()
if (failures.length) { for (const f of failures) console.error(`  FAIL  ${f}`); process.exit(1) }
console.log('  ok    the banner matches this server, route for route')
