/**
 * Q140: the thumbnail that has not arrived yet must hold its own space.
 *
 * Q127 routed episode thumbnails through an authed fetch because a broken <img> reads as "the recording
 * captured nothing" — the worst possible lie right after a take. But the LOADING state said the same
 * thing: `.thumb-loading` had no rule anywhere, so an empty <span> is 0x0 and the episode showed three
 * nothings in a 0-height row before the pictures popped in and shoved the layout. Over the tunnel, where
 * every one of those bytes travels with a bearer token, that window is longest.
 *
 * Only a browser can answer this, and it needs the bytes HELD: the subject is the state between request
 * and arrival. The thumbnail route is gated by this script, so the placeholder is measured while it is
 * genuinely pending, then released and the SAME slot measured again — a placeholder that reserves the
 * wrong box is a layout shift, which is a different defect from reserving none.
 *
 * Run: node scripts/audit-thumb-placeholder.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

// A VALID 1x1 PNG. It must really decode: an undecodable image is sized by its ALT TEXT instead
// (measured, 68x72 — wider than tall, ignoring the CSS aspect-ratio entirely), and this audit then
// reports a layout shift that belongs to its own fixture rather than to the page. The intrinsic size does
// not matter because `.rec-thumbs img` sets aspect-ratio itself; being decodable does.
const PIXEL = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==',
  'base64')

const session = {
  dataset: 'cagatay/so101-pick', task: 'pick up the cube', leader: 'so101-arm-1', follower: 'so101-arm-2',
  target_episodes: 3, phase: 'idle', fps: 30, fps_achieved: 29.8, fps_notice: null, camera_notice: null,
  error: null, motion_notice: null,
  episodes: [{ index: 0, frames: 354, duration_s: 11.8, status: 'done', thumbnails: {
    top: '/api/record/thumb/0/top', wrist: '/api/record/thumb/0/wrist', side: '/api/record/thumb/0/side' } }],
}

const browser = await chromium.launch()
// serviceWorkers:'block' is REQUIRED: a response served by this PWA's service worker is not
// interceptable by page.route, and the injected session would silently never land.
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
const thrown = []
page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))

// ORDER MATTERS: playwright tries the LAST registered matching handler first, so the broad guards go
// first and the specific thumbnail gate last. Registered the other way round the catch-all answers the
// thumbnail with JSON, the placeholder never appears, and this audit reports "no placeholder" — blaming
// the UI for its own plumbing. (Cost this script its first run.)
await page.route('**/api/record/session', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(session) }))
await page.route('**/api/record/**', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(session) }))
let release = false
const pending = []
await page.route('**/api/record/thumb/**', async r => {
  if (!release) { pending.push(r); return }             // held: the subject is the pending state
  await r.fulfill({ status: 200, contentType: 'image/png', body: PIXEL })
})

await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(5000)
await page.locator('button.chip:has-text("record")').first().click()
await page.waitForTimeout(2500)

const before = await page.evaluate(() => {
  const ph = [...document.querySelectorAll('.thumb-loading')]
  const row = document.querySelector('.rec-thumbs')?.getBoundingClientRect()
  return { n: ph.length, row: row ? Math.round(row.height) : 0,
    boxes: ph.map(e => { const b = e.getBoundingClientRect(); return { w: Math.round(b.width), h: Math.round(b.height) } }) }
})
if (!before.n) {
  failures.push('no .thumb-loading placeholder while the thumbnail bytes are held — the loading state '
    + 'cannot be checked, so either AuthedImg changed or the fixture never landed')
} else {
  const flat = before.boxes.filter(b => b.w < 8 || b.h < 8)
  console.log(`  pending: ${before.n} placeholder(s) ${before.boxes.map(b => `${b.w}x${b.h}`).join(' ')} `
    + `in a ${before.row}px row`)
  if (flat.length) failures.push(`${flat.length} of ${before.n} thumbnail placeholders are effectively `
    + `invisible (${flat.map(b => `${b.w}x${b.h}`).join(', ')}) — while the bytes are in flight the episode `
    + 'shows nothing where its pictures go, which reads as "the recording captured nothing" (the exact '
    + 'misreading Q127 fixed for the FAILED case) and then shifts the layout when they arrive')
}

// Release the same three requests and measure the slot again: reserving the WRONG box is a layout shift.
release = true
for (const r of pending) await r.fulfill({ status: 200, contentType: 'image/png', body: PIXEL })
await page.waitForTimeout(1500)
const after = await page.evaluate(() => {
  const im = [...document.querySelectorAll('.rec-thumbs img')]
  return { n: im.length, boxes: im.map(e => { const b = e.getBoundingClientRect(); return { w: Math.round(b.width), h: Math.round(b.height) } }) }
})
if (!after.n) {
  failures.push('the thumbnails never became <img> after their bytes arrived — AuthedImg is stuck on the placeholder')
} else if (before.n) {
  console.log(`  arrived: ${after.n} image(s) ${after.boxes.map(b => `${b.w}x${b.h}`).join(' ')}`)
  const shift = before.boxes.slice(0, after.n).map((b, i) =>
    Math.abs(b.w - after.boxes[i].w) + Math.abs(b.h - after.boxes[i].h)).filter(d => d > 2)
  if (shift.length) failures.push(`the placeholder does not reserve the picture's box (${shift.length} of `
    + `${after.n} differ: ${before.boxes.map(b => `${b.w}x${b.h}`).join(' ')} -> `
    + `${after.boxes.map(b => `${b.w}x${b.h}`).join(' ')}) — the row still jumps when a thumbnail lands`)
}
if (thrown.length) failures.push(`page errors: ${thrown.join(' | ')}`)

await ctx.close(); await browser.close()
if (failures.length) { for (const f of failures) console.error(`  FAIL  ${f}`); process.exit(1) }
console.log('thumb placeholder: a thumbnail in flight holds exactly the box its picture will occupy')
