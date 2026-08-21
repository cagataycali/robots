/**
 * Q135: THE FIRST RUN — what a person sees before any robot, profile or dataset exists.
 *
 * Every other audit runs against this rig, which always has peers, two servo boards, four cameras and
 * saved profiles. So the one state NO audit covered is the state EVERY new user starts in, and an empty
 * screen is where a product either explains itself or loses the person: "0 peers" and a blank panel are
 * indistinguishable from a broken dashboard.
 *
 * The fixture is the interesting part. The fleet does NOT arrive over /api/fleet — it streams over
 * /ws/mesh (useMesh.ts), so an "empty machine" built by routing HTTP alone is a FICTION: the page went
 * on showing this rig's two real arms while the audit believed it was testing a fresh install. Two
 * traps, both cost a run: the PWA service worker answers cached API calls unless serviceWorkers is
 * blocked, and the socket URL carries ?token=, so the glob '**\/ws\/mesh' never matches (a regex does).
 *
 * What it demands of every screen in that state: a NEXT ACTION in words, and no placeholder leaking
 * through (undefined/NaN/[object Object]) — the failure mode of an empty payload meeting a template.
 */
import fs from 'node:fs'
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
page.on('pageerror', e => failures.push(`page threw on a fresh install: ${String(e.message).slice(0, 140)}`))

await page.routeWebSocket(/\/ws\/mesh/, ws => {
  ws.send(JSON.stringify({ type: 'snapshot', dashboard_peer_id: 'dashboard-fresh', peers: {},
    mesh: { online: true }, t: Date.now() / 1000 }))
})
const empty = {
  '**/api/devices': { serial_ports: [], cameras: [], camera_problem: null, camera_names: [], managed: {} },
  '**/api/devices/profiles': { profiles: {}, path: '/tmp/none.json', autospawn: false },
  // The train screen's list comes from /api/training/datasets (NOT /api/datasets — that path only
  // serves episode labels). A pattern that matches nothing fails silently: the first version of this
  // audit "proved" the empty train screen while the page was still listing a real dataset off this
  // machine. Emptiness fixtures must be checked against the payload, which is why the assertions
  // below look for the words, not just the absence of rows.
  '**/api/training/datasets*': { datasets: [] },
  '**/api/training/trainers*': { trainers: [] },
  '**/api/datasets/labels*': { labels: {} },
  '**/api/checkpoints*': { checkpoints: [] },
  '**/api/fleet*': { type: 'snapshot', dashboard_peer_id: 'dashboard-fresh', peers: {}, mesh: { online: true }, t: 0 },
}
for (const [pattern, json] of Object.entries(empty)) await page.route(pattern, r => r.fulfill({ json }))

await page.goto(`${BASE}/?token=${encodeURIComponent(TOKEN)}`, { waitUntil: 'domcontentloaded' })
await page.waitForTimeout(4500)

/** The text of the layer the person is actually looking at: an open panel, else the page. */
const visibleText = () => page.evaluate(() => {
  const layers = [...document.querySelectorAll('.train-sheet, .drawer, .sheet-backdrop')]
    .filter(el => el.getBoundingClientRect().height > 100)
  const el = layers.length ? layers[layers.length - 1] : (document.querySelector('.app') ?? document.body)
  return el.innerText.replace(/\n{2,}/g, '\n').trim()
})

// The fixture must be PROVEN to have landed, or every assertion below is about the wrong machine.
const head = await visibleText()
if (!/0 peers/.test(head) || !/dashboard-fresh/.test(head)) {
  failures.push('THE FIXTURE DID NOT LAND: the page still shows this rig, not a fresh install '
    + `(header says "${head.split('\n').slice(0, 3).join(' / ')}") — check the /ws/mesh route and serviceWorkers:'block'`)
}

for (const [name, chip] of [['fleet', null], ['record', 'record'], ['train', 'train'], ['devices', 'devices']]) {
  if (chip) {
    await page.locator(`button.chip:has-text("${chip}")`).first().click()
    await page.waitForTimeout(2600)
  }
  const text = await visibleText()
  const say = t => t.split('\n').filter(l => l.trim().length > 12).slice(0, 4).join(' ⋅ ')
  console.log(`  ${name}: ${say(text).slice(0, 210)}`)

  for (const junk of ['undefined', 'NaN', '[object Object]', 'null']) {
    if (new RegExp(`(^|\\s)${junk.replace(/[[\]]/g, '\\$&')}(\\s|$)`).test(text))
      failures.push(`${name}: a fresh install leaks "${junk}" into the page — an empty payload met a template`)
  }
  // A NEXT ACTION, not just an absence. "No datasets" is a fact; "record one on the record screen" is a product.
  const imperative = /\b(plug|spawn|start|record|pick|choose|select|connect|click|run|add|calibrate|create|collect|open)\b/i
  if (!imperative.test(text)) failures.push(`${name}: nothing on the empty screen tells the person what to do next`)
  if (text.replace(/\s/g, '').length < 40) failures.push(`${name}: the empty screen is blank — indistinguishable from a broken dashboard`)
  if (chip) { await page.keyboard.press('Escape').catch(() => {}); await page.waitForTimeout(700) }
}

await browser.close()
if (failures.length) { console.error('FAIL\n' + failures.map(f => ` - ${f}`).join('\n')); process.exit(1) }
console.log('first run: with no peers, no boards, no profiles and no datasets, every screen still says what '
  + 'to do next — and no empty payload leaks a placeholder into the page')
