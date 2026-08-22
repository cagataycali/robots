/**
 * U22 slice 3b: the one button on this dashboard whose effect is AN ARM IN MOTION.
 *
 * Every request is fixtured — the fleet, the teleop status, and both start routes — so this audit cannot
 * move, publish or command anything. The guarded browser (lib/audit-browser.mjs) blocks anything mutating
 * I failed to claim and fails the process if so, which is the only reason it is safe to audit this button
 * at all: the first version of the stop audit escaped through a glob and sent two real commands.
 *
 * What only a browser can answer: does the offer appear ONLY when an arm can actually lead, does ONE click
 * refuse to start, does the confirm sentence name WHICH ARM MOVES (an operator standing between two arms
 * cannot act on "start teleop"), is the leader made to publish BEFORE the follower is pointed at it, and —
 * the assertion this arc exists for — is "started, but every frame refused" reported as a FAILURE naming
 * the grant that widens the bound, rather than as a working session?
 */
import { chromium } from './lib/audit-browser.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(`${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []
/* The real payload's shape, read off the running server: /api/fleet returns `type` + `peers` as a DICT
   KEYED BY PEER ID (not a list — a fixture built as a list renders nothing and the audit then blames the
   UI), each peer carrying state/presence/stale plus the measured role fields at the top level. */
const peer = (id, joints, extra = {}) => ({
  peer_id: id, last_seen: Date.now() / 1000, stale: false, presence: { connected: true }, cameras: {},
  state: { peer_id: id, t: Date.now() / 1000, connected: true,
    joints: Object.fromEntries(Array.from({ length: joints }, (_, i) => [`j${i}`, 0.1])) },
  role: null, role_volts: null, role_source: null, ...extra,
})
/* The frame useMesh actually accepts is type:'snapshot' — NOT 'fleet' (its switch drops every unknown
   type silently, which is why three runs rendered an empty fleet and looked like a UI defect). `t` is the
   SERVER's clock: mergeMeshEvent rebases each peer's last_seen by AGE into browser time, so a fixture must
   send both or every peer renders stale. */
const fleetDoc = list => ({ type: 'snapshot', dashboard_peer_id: 'dash', mesh: { online: true },
  t: Date.now() / 1000, peers: Object.fromEntries(list.map(p => [p.peer_id, p])),
  absent_children: [], managed_no_presence: [] })
const IDLE = { health: { receivers: {}, publishers: {} } }
const REFUSING = { health: { worst: { state: 'refusing', headline: 'every frame is being refused', refusal: 'out of range' } } }
const FOLLOWING = { health: { worst: { state: 'following', headline: 'following lead-arm at 9.8Hz' } } }

const browser = await chromium.launch()
const open = async ({ fleet, afterStart }) => {
  const ctx = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1280, height: 1000 } })
  const page = await ctx.newPage()
  /* THE FLEET SCREEN IS FED BY THE /ws/mesh WEBSOCKET, not only by the /api/fleet poll: a fixtured poll is
     overwritten by the first live frame a second later, and the audit then looks like a UI defect (it cost
     this audit two runs). Swallowing the socket — accept it, deliver nothing — leaves the fixtured
     snapshot standing, and is also the safer posture: no live peer can enter a scenario about arms moving. */
  /* THE FLEET COMES ONLY FROM /ws/mesh (lib/useMesh.ts: "one WebSocket → normalized reactive fleet
     store"), so a fixtured /api/fleet poll renders NOTHING and the audit reads like a UI defect — it cost
     this audit three runs. The socket is therefore MOCKED, not swallowed: playwright answers it and this
     audit is the only thing that can put a peer on the screen, which also means no live arm can wander
     into a scenario about arms moving. */
  await page.routeWebSocket(/\/ws\/mesh/, ws => {
    const send = () => ws.send(JSON.stringify(fleetDoc(fleet)))
    send()
    /* Re-sent because a snapshot ages: peers rendered from one frame go stale while a scenario runs.
       UNREF'd because a repeating timer keeps node's event loop open — this audit PASSED and then hung
       forever after browser.close(), which in run-audits would stall the whole suite behind a green
       result. An audit that cannot exit has not finished. */
    const t = setInterval(send, 1000)
    t.unref?.()
    ws.onClose(() => clearInterval(t))
  })
  const calls = []
  let asked = 0
  await page.route(/\/api\/fleet(\?|$)/, r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(fleetDoc(fleet)) }))
  // Regex, never a glob: playwright's '*' does not cross a '/', which is how two real stop commands escaped.
  await page.route(/\/api\/robots\/[^/]+\/teleop(\/.*)?$/, r => {
    const req = r.request()
    const path = new URL(req.url()).pathname
    if (req.method() === 'POST') { calls.push(path)
      return r.fulfill({ status: 200, contentType: 'application/json', body: '{"result":{"ok":true}}' }) }
    asked += 1
    const body = calls.length ? afterStart : IDLE
    return r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
  })
  await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(2600)
  const card = page.locator(`text=${fleet[0].peer_id}`).first()
  if (await card.count()) await card.click()
  await page.waitForTimeout(900)
  const tbtn = page.locator('button', { hasText: /^teleop$/ }).first()
  if (await tbtn.count()) await tbtn.click()
  await page.waitForTimeout(1200)
  return { page, ctx, calls, asked: () => asked }
}
const teleopText = async page =>
  (await page.locator('div.hint').filter({ hasText: 'teleop' }).first().innerText().catch(() => '')).replace(/\s+/g, ' ')

// ---- 1. THE REAL FLEET SHAPE: a jointless leader can lead nothing, so there is no button to press.
{
  const { page, ctx, calls } = await open({ fleet: [peer('so101-follower', 6), peer('so101-leader', 0)], afterStart: IDLE })
  const t = await teleopText(page)
  if (!/no arm on this fleet can lead it yet/.test(t)) failures.push(`a jointless fleet did not explain itself: ${t.slice(0, 140)}`)
  if (await page.locator('button', { hasText: /^follow / }).count()) failures.push('offered to follow an arm that reports no joints')
  if (calls.length) failures.push('something was started without a click')
  await ctx.close()
}

// ---- 2. AN ARM THAT CAN LEAD: offered, armed, and only then started — publish before receive.
{
  const { page, ctx, calls } = await open({ fleet: [peer('so101-follower', 6), peer('lead-arm', 6, { role: 'leader', role_volts: 7.4, role_source: 'measured' })], afterStart: FOLLOWING })
  const offer = page.locator('button', { hasText: /^follow lead-arm$/ }).first()
  if (!await offer.count()) { failures.push('an arm reporting 6 joints was not offered as a leader') }
  else {
    const t = await teleopText(page)
    if (!/measured as a leader \(7\.4V\)/.test(t)) failures.push('the measured role evidence is not on the screen')
    if (!/agent_physical_motion/.test(t) || !/teleop_degree_units/.test(t)) failures.push('the grants a start needs are not named before the click')
    await offer.click(); await page.waitForTimeout(400)
    if (calls.length) failures.push('the FIRST click started teleop — an arm must not move on one click')
    const confirm = page.locator('button', { hasText: /^confirm — hand-guide/ }).first()
    if (!await confirm.count()) failures.push('no confirm step')
    else {
      const label = await confirm.innerText()
      if (!/lead-arm/.test(label) || !/so101-follower MOVES/.test(label))
        failures.push(`the confirm sentence does not say which arm moves: ${label}`)
      if (!await page.locator('button', { hasText: /^cancel$/ }).count()) failures.push('the armed state has no way back')
      await confirm.click(); await page.waitForTimeout(2000)
      if (calls.length !== 2) failures.push(`expected exactly 2 requests (publish then receive), got ${calls.length}: ${calls.join(', ')}`)
      else if (!/lead-arm\/teleop\/publish$/.test(calls[0]) || !/so101-follower\/teleop\/receive$/.test(calls[1]))
        failures.push(`wrong order — the follower must not be pointed at a stream nobody publishes: ${calls.join(' then ')}`)
      const after = await teleopText(page)
      if (!/teleop live/.test(after)) failures.push(`a working session was not reported as live: ${after.slice(0, 160)}`)
      else console.log(`  note  a start that took says: ${after.match(/teleop live[^·]*/)?.[0]?.slice(0, 90)}`)
    }
  }
  await ctx.close()
}

// ---- 3. THE OUTCOME THIS FLEET HAS ACTUALLY PRODUCED: "receive started", every frame refused.
{
  const { page, ctx } = await open({ fleet: [peer('so101-follower', 6), peer('lead-arm', 6)], afterStart: REFUSING })
  await page.locator('button', { hasText: /^follow lead-arm$/ }).first().click()
  await page.waitForTimeout(300)
  await page.locator('button', { hasText: /^confirm — hand-guide/ }).first().click()
  await page.waitForTimeout(2000)
  const t = await teleopText(page)
  if (/teleop live/.test(t)) failures.push(`a stream refusing every frame was reported as live: ${t.slice(0, 160)}`)
  if (!/REFUSED/.test(t)) failures.push(`the refusal is not stated after a start: ${t.slice(0, 160)}`)
  if (!/teleop_degree_units/.test(t)) failures.push('the refusal does not name the grant that widens the bound')
  else console.log(`  note  a start whose frames are refused says: ${t.match(/started, but[^·]*/)?.[0]?.slice(0, 120)}`)
  await ctx.close()
}

await browser.close()
if (failures.length) { for (const f of failures) console.error(`  FAIL  ${f}`); process.exit(1) }
console.log('  ok    an arm is offered only when it can lead, one click never moves anything, publish precedes receive, and a refused stream is not a session')
