/**
 * A grant you cannot see is a grant you cannot take back (Q119 → Q121).
 *
 * The two policy allowlists added by Q119 are server-side facts: granted_state returns them,
 * revoke_patch handles them, /api/consent/revoke is KINDS-driven. The unit tests prove all of that.
 * The part only a browser can answer is whether the PERMISSIONS SCREEN renders them and posts the
 * right kind when the operator clicks revoke — and whether the "nothing is allowed here" assurance
 * stays silent while grants exist (Q121, a sentence that was wrong three times).
 *
 * /api/consent is INJECTED because the running server predates Q119 (its `kinds` has three entries),
 * and the revoke POST is intercepted and RECORDED, never forwarded: this audit must not change what
 * cagatay's machine actually permits.
 *
 * Two recorded laws for this repo are load-bearing here: the dashboard is a PWA, so the context must
 * block service workers or an injected response never lands; and page.route matches in REVERSE
 * registration order, so the specific routes are registered LAST.
 *
 * Run: node scripts/audit-consent-rows.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []
const ok = (cond, what) => { console.log(`${cond ? '  ok  ' : '  FAIL'} ${what}`); if (!cond) failures.push(what) }

/* Exactly the shape strands_robots.dashboard.consent.granted_state returns for a machine with the
   two Q119 grants in force (verified by piping that function's JSON through the bundle). `kinds` is
   deliberately present and non-empty: it is metadata, and Q121's "nothing" rule must not count it. */
const CONSENT = {
  kinds: ['trust_remote_code', 'hf_repo_allow', 'teleop_degree_units', 'agent_physical_motion',
    'policy_type_allow', 'policy_host_allow'],
  trust_remote_code: false,
  agent_physical_motion: false,
  hf_repo_allow: [],
  policy_type_allow: ['smolvla_x'],
  policy_host_allow: ['gpu.lan', '10.0.0.0/24'],
  teleop_degree_units: { granted: false, value_abs: null, slew_abs: null, is_degree_preset: false },
  locks: { task_requires_confirm: false, task_requires_confirm_env: 'STRANDS_DASH_TASK_REQUIRES_CONFIRM' },
  env_file: '.env',
}

const browser = await chromium.launch()
// PWA: without this the service worker answers /api/consent from its cache and the injection is a no-op.
const context = await browser.newContext({ serviceWorkers: 'block' })
const page = await context.newPage()
const revokePosts = []
let consentPayload = CONSENT

// Catch-all FIRST (reverse-order law): anything not named below goes to the real server.
await page.route('**/api/**', route => route.continue())
await page.route('**/api/consent', route => route.fulfill({
  status: 200, contentType: 'application/json', body: JSON.stringify(consentPayload),
}))
await page.route('**/api/consent/revoke', route => {
  revokePosts.push(JSON.parse(route.request().postData() ?? '{}'))
  const body = JSON.parse(route.request().postData() ?? '{}')
  // Answer as the server would, and drop the entry so the reload proves the row disappears.
  consentPayload = {
    ...consentPayload,
    policy_type_allow: consentPayload.policy_type_allow.filter(e => e !== body.subject),
    policy_host_allow: consentPayload.policy_host_allow.filter(e => e !== body.subject),
  }
  route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ revoked: true }) })
})

await page.goto(`${BASE}/?token=${encodeURIComponent(TOKEN)}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(1200)
await page.getByTitle('Settings').click()
const consented = page.waitForResponse(r => r.url().includes('/api/consent') && r.status() === 200)
await page.getByRole('button', { name: /security/i }).click()
await consented  // the rows render from THIS response — reading before it lands invents a defect (recorded law)
await page.waitForTimeout(400)

const body = await page.locator('.consent-settings').innerText()
ok(/smolvla_x/.test(body), 'the granted policy type is listed')
ok(/gpu\.lan/.test(body), 'the granted policy host is listed')
ok(/10\.0\.0\.0\/24/.test(body), 'a CIDR grant is listed')
ok(/range/i.test(body), 'the CIDR row says it is wider than one host')
ok(/camera frames|joint states/i.test(body), 'the host row says what that machine receives')
ok(!/Nothing extra is allowed here/i.test(body),
  'Q121: no false assurance while grants are in force')

/* Count only rows with a REVOKE button. My first draft asserted "three .cg-row" and failed on four:
   the fourth is the Q81 LOCKS row ("Require the ▶ confirmation before real motion"), which always
   renders and is a restriction, not a grant. Same distinction Q121 encodes in NOT_A_GRANT — an
   audit that conflates them reports a defect that does not exist. */
const grantRows = page.locator('.consent-settings .cg-row', { has: page.getByRole('button', { name: /revoke/i }) })
ok(await grantRows.count() === 3, `three revocable grant rows (got ${await grantRows.count()})`)

// Revoking must post the KIND, not just the value: the endpoint rebuilds the request from it.
await page.locator('.cg-row', { hasText: 'gpu.lan' }).getByRole('button', { name: /revoke/i }).click()
await page.waitForTimeout(900)
ok(revokePosts.length === 1, `one revoke posted (got ${revokePosts.length})`)
ok(revokePosts[0]?.kind === 'policy_host_allow', `kind is policy_host_allow (got ${revokePosts[0]?.kind})`)
ok(revokePosts[0]?.subject === 'gpu.lan', `subject is the entry (got ${revokePosts[0]?.subject})`)
/* Asserted on the ROW, not the panel text: the panel legitimately still says "revoked gpu.lan" in
   its confirmation note, and reading the whole panel made a working revocation look broken. */
ok(await page.locator('.cg-row', { hasText: 'gpu.lan' }).count() === 0, 'the revoked host row is gone')
ok(await page.locator('.consent-settings').innerText().then(t => /revoked gpu\.lan/.test(t)),
  'and the screen confirms what it revoked, by name')
ok(await page.locator('.cg-row', { hasText: 'smolvla_x' }).count() === 1, 'the other grants are untouched')
ok(await page.locator('.cg-row', { hasText: '10.0.0.0/24' }).count() === 1, 'including the CIDR entry')

// And the assurance must appear when there is genuinely nothing.
consentPayload = { ...CONSENT, policy_type_allow: [], policy_host_allow: [] }
await page.reload({ waitUntil: 'domcontentloaded' })
await page.waitForTimeout(1200)
await page.getByTitle('Settings').click()
await page.getByRole('button', { name: /security/i }).click()
await page.waitForTimeout(700)
const empty = await page.locator('.consent-settings').innerText()
ok(/Nothing extra is allowed here/i.test(empty), 'the assurance DOES appear when nothing is granted')

await browser.close()
console.log(failures.length ? `\nFAIL ${failures.length}: ${failures.join(' | ')}` : '\nPASS consent rows are visible and revocable')
process.exit(failures.length ? 1 : 0)
