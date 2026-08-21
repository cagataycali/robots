#!/usr/bin/env node
/**
 * Q103 in a real browser: a token that goes INVALID while the page is open must bring the gate back.
 *
 * Simulates the actual failure — a dashboard restart rotates local_api_token.txt, or a session is
 * revoked — by replacing the stored token WITHOUT setAuthToken (which would remount the app and prove
 * nothing). The page keeps its React tree, its pollers start getting 401s, and the question is whether
 * anything on screen ever says so.
 *
 * Read-only against the live dashboard: no arm moves, no camera socket is opened by this script.
 */
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations
import fs from 'node:fs'

const TOKEN = fs.readFileSync(process.env.HOME + '/.strands_dashboard/local_api_token.txt', 'utf8').trim()
const BASE = process.env.BASE ?? 'http://127.0.0.1:8090'

const browser = await chromium.launch()
// A service worker would serve the API from cache and hide the refusal entirely (a law of this repo).
const ctx = await browser.newContext({ serviceWorkers: 'block' })
const page = await ctx.newPage()

await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForResponse(r => r.url().includes('/api/fleet') && r.status() === 200, { timeout: 20_000 })
const openedWith = await page.evaluate(() => document.body.innerText.slice(0, 80).replace(/\n/g, ' | '))
console.log(`opened: ${openedWith}`)

// The rotation. Well-formed enough to be sent, invalid to the server, and NOT expired — so
// lib/sessionExpiry has nothing to complain about and only the server can tell.
await page.evaluate(() => localStorage.setItem('strands.token', 'rotated-by-a-restart-not-expired'))

page.on('console', m => { if (/error|warn/i.test(m.type())) console.log('  console:', m.text().slice(0, 140)) })
page.on('pageerror', e => console.log('  PAGEERROR:', String(e).slice(0, 200)))
const calls = []
page.on('request', r => { if (r.url().includes('/api/')) calls.push(r.url().split('/api/')[1].split('?')[0]) })
page.on('requestfailed', r => { if (r.url().includes('/api/')) console.log('  REQUESTFAILED:', r.url().split('/api/')[1].split('?')[0], r.failure()?.errorText) })
let firstRefusalAt = null
const rotatedAt = Date.now()
const refusals = []
page.on('response', r => {
  if (!r.url().includes('/api/')) return
  const tag = `${r.url().split('/api/')[1].split('?')[0]}=${r.status()}${r.fromServiceWorker() ? '(sw)' : ''}`
  if ((r.status() === 401 || r.status() === 403) && firstRefusalAt === null) firstRefusalAt = Date.now()
  refusals.push(tag)
})

// Nudge the watcher the way a returning phone would, then let its 30s timer be irrelevant.
await page.evaluate(() => window.dispatchEvent(new Event('focus')))
let text = ''
const t0 = Date.now()
for (let i = 0; i < 75; i++) {
  await page.waitForTimeout(1000)
  await page.evaluate(() => window.dispatchEvent(new Event('focus')))
  text = await page.evaluate(() => document.body.innerText)
  if (/not signed in any more|passkey/i.test(text)) { console.log(`  gate returned after ${((Date.now() - t0) / 1000).toFixed(1)}s`); break }
}

console.log(`  first guarded refusal: ${firstRefusalAt === null ? 'NONE in the window' : ((firstRefusalAt - rotatedAt) / 1000).toFixed(1) + 's after the rotation'}`)
console.log('  api calls after rotation:', [...new Set(calls)].join(', ').slice(0, 200))
console.log(`refused calls seen: ${refusals.length ? refusals.slice(0, 4).join(', ') : 'none'}`)
const gated = /not signed in any more/i.test(text)
console.log(gated ? 'PASS  the gate came back and said why:' : 'FAIL  the page never admitted the refusal:')
console.log('      ' + text.split('\n').filter(Boolean).slice(0, 6).join(' | ').slice(0, 300))
await browser.close()
process.exit(gated ? 0 : 1)
