/**
 * Does the security screen SAY that the token is readable via `ps`? (commit 40b68667's UI half)
 *
 * The backend half is unit-tested, but the whole point of the notice is a sentence a human reads on
 * a screen, and this dashboard has already shipped three rules that were correct, tested, green and
 * never rendered. So this drives the real page.
 *
 * Both halves matter and the second is the one that keeps the screen honest:
 *   present  -> the exposure AND the remedy flag are on screen, under the green auth tick
 *   absent   -> nothing at all is added (an older server omits the field; "probably fine" invented
 *               from silence is how a security screen starts lying)
 */
import fs from 'node:fs'
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations

const BASE = process.env.DASH || 'http://127.0.0.1:8090'
const TOKEN = fs.readFileSync(`${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const SECRET = 'kDD6toTMVDwOXYn51XfDI0vNnKGC4tSM'

const NOTICE = {
  kind: 'token_in_argv',
  severity: 'warn',
  text: 'This dashboard was started with --auth-token on the command line, so its bearer token is readable by every local user via `ps` - and that token is what stops a stranger on this machine from driving the arms.',
  remedy: 'Next restart, pass --auth-token-file ~/.strands_dashboard/local_api_token.txt instead: same token, same auth, out of argv and out of shell history. Rotate the token if this machine is shared.',
}

async function readSecurityTab(page, notice) {
  // The PWA's service worker would serve /api/config from its own cache and this fixture would
  // never land — the trap that once made an audit report "the UI shows nothing" about itself.
  await page.route('**/api/config', async route => {
    const res = await route.fetch()
    let doc = {}
    try { doc = await res.json() } catch { doc = {} }
    doc.security = { ...(doc.security || {}), auth_enabled: true, ...(notice ? { notice } : {}) }
    if (!notice) delete doc.security.notice
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(doc) })
  })
  await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(3500)
  const gear = page.locator('button[title*="ettings"], button[aria-label*="ettings"], button.chip:has-text("settings")')
  await gear.first().click()
  await page.waitForTimeout(1200)
  await page.locator('button:has-text("Security")').first().click()
  await page.waitForTimeout(900)
  return (await page.locator('.drawer-body, .drawer, section').first().innerText()).replace(/\s+/g, ' ')
}

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1000 }, serviceWorkers: 'block' })
const fails = []

const withNotice = await readSecurityTab(await ctx.newPage(), NOTICE)
for (const needle of ['readable by every local user', '--auth-token-file', 'ps']) {
  const ok = withNotice.includes(needle)
  console.log(`  ${ok ? 'ok   ' : 'FAIL '} security tab says ${JSON.stringify(needle)}`)
  if (!ok) fails.push(`missing on screen: ${needle}`)
}
if (withNotice.includes(SECRET)) fails.push('the screen printed the token itself')

const withoutNotice = await readSecurityTab(await ctx.newPage(), null)
const quiet = !withoutNotice.includes('--auth-token-file') && !withoutNotice.includes('readable by every local user')
console.log(`  ${quiet ? 'ok   ' : 'FAIL '} a server that sends no notice adds nothing`)
if (!quiet) fails.push('invented a warning from an absent field')

await browser.close()
console.log(fails.length ? `FAIL  ${fails.join(' | ')}` : 'PASS  the security screen names the argv exposure and its remedy, and stays quiet without one')
process.exit(fails.length ? 1 : 0)
