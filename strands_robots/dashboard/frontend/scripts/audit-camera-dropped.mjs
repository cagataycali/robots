/**
 * The `cameras dropped` verdict RENDERS — proven on the page, with the socket the UI actually reads.
 *
 * Why this script exists at all: the U2 bug was a route and a websocket telling different stories, and I
 * "verified" that fix on the endpoint I had changed while the badge never appeared on screen. The fleet view
 * renders from /ws/mesh, so a page-level proof has to arrive through /ws/mesh. playwright's routeWebSocket
 * lets us serve a synthetic snapshot WITHOUT touching the live dashboard (which is running a backend that
 * predates the cameras_requested annotation, and which must never be restarted from a background loop —
 * camera TCC law).
 *
 * Two specimens, both real shapes measured on this rig:
 *   dropped-arm : spawned WITH top+wrist, announces none  -> "cameras dropped", a cause, and a log to read
 *   mute-arm    : announces `top`, no frames arriving      -> must stay "no frames" (Q45's dead reader
 *                 thread, NOT a permission — blaming TCC here sends the operator to the wrong screen)
 *
 * Reads only. No robot is commanded; nothing is sent to the real mesh.
 * Run: node scripts/audit-camera-dropped.mjs
 */
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`,
  'utf8',
).trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const now = Date.now() / 1000

const peer = (id, presence, extra) => ({
  peer_id: id, last_seen: now, stale: false, cameras: {},
  presence: { connected: true, hostname: 'Mac', robot_type: 'so101', timestamp: now, ...presence },
  state: { 'shoulder_pan.pos': 1.5, 'gripper.pos': 12 },
  ...extra,
})

const SNAPSHOT = {
  type: 'snapshot',
  dashboard_peer_id: 'gateway-audit',
  mesh: { connected: true },
  t: now,
  peers: {
    // the arm-2 shape: the dashboard asked for two cameras, the robot announces none
    'dropped-arm': peer('dropped-arm', {}, { cameras_requested: ['top', 'wrist'], origin: 'managed' }),
    // the Q45 shape: the camera opened and then went quiet
    'mute-arm': peer('mute-arm', { cameras: ['top'] }, { cameras_requested: ['top'], origin: 'managed' }),
  },
}

const fails = []
const check = (ok, label, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) fails.push(label)
}

const browser = await chromium.launch()
// The dashboard is a PWA: a service-worker response is not interceptable, and an injected fixture
// silently never lands (the audit then blames the UI for a fixture that never arrived).
const ctx = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1280, height: 900 } })
const page = await ctx.newPage()

// Only the mesh socket is mocked; camera/chat sockets keep their real behaviour.
await page.routeWebSocket(/\/ws\/mesh/, ws => {
  ws.onMessage(() => {})               // the client sends keepalives; swallow them
  ws.send(JSON.stringify(SNAPSHOT))
})

page.on('pageerror', e => check(false, 'page threw', e.message))
await page.goto(`${BASE}/?token=${encodeURIComponent(TOKEN)}`, { waitUntil: 'domcontentloaded' })
await page.getByText('dropped-arm').first().waitFor({ timeout: 15000 })
check(true, 'the mocked snapshot reached the page (both specimens rendered)')

// --- the card must NOT shout about it: silence is the honest option on a grid ---
const cardText = await page.locator('.card', { hasText: 'dropped-arm' }).first().innerText()
check(!/cameras dropped/i.test(cardText), 'the fleet card stays quiet about missing cameras')

// --- the detail stage is where the question gets answered ----------------------
await page.getByText('dropped-arm').first().click()
const stage = page.locator('.detail-stage .camstate').first()
await stage.waitFor({ timeout: 10000 })
const head = (await stage.locator('b').innerText()).trim()
const sub = (await stage.locator('span').innerText()).trim()
const title = (await stage.getAttribute('title')) ?? ''

check(head === 'cameras dropped', 'stage head names the failure, not the hardware', `head=${JSON.stringify(head)}`)
check(/top, wrist requested, none opened/.test(sub), 'stage sub names the cameras we asked for', sub)
check(!/publishes none|lists no cameras/i.test(`${head} ${sub} ${title}`), 'the old denial is gone from the page')
check(/dropped when it connected/.test(title), 'the full sentence is reachable (title)')
check(/blocked by macOS privacy|another process|unplugged/.test(title), 'the causes are named, none is picked')
check(/devices › logs/.test(title), 'it points at the log that names the failing camera')
check(/would capture joints only/.test(title), 'conditional: nothing is recording yet')

// --- the mute specimen must not be relabelled as a permission problem ----------
await page.keyboard.press('Escape')
await page.getByText('mute-arm').first().click()
const stage2 = page.locator('.detail-stage .camstate').first()
await stage2.waitFor({ timeout: 10000 })
const head2 = (await stage2.locator('b').innerText()).trim()
const title2 = (await stage2.getAttribute('title')) ?? ''
check(head2 === 'no frames', 'an announced-but-silent camera stays "no frames"', `head=${JSON.stringify(head2)}`)
check(!/dropped when it connected/.test(title2), 'it is NOT blamed on a permission (Q45 is a dead reader thread)')

await page.screenshot({ path: '/tmp/camera_dropped_audit.png' })
await browser.close()
console.log(fails.length ? `  FAIL  ${fails.length} check(s): ${fails.join(' | ')}` : '  PASS  the dropped/mute verdicts render on the page')
process.exit(fails.length ? 1 : 0)
