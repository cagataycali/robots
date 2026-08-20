/**
 * Q48: the training form must not offer a provider it cannot submit.
 *
 * `ppo` and `fast_sac` need an RLTrainSpec built in a script; the form builds a supervised TrainSpec
 * by construction, so picking one used to cost a dataset choice and a click to earn
 * "ppo requires an RLTrainSpec, got TrainSpec" — a sentence about internal classes, on a path that
 * could never succeed. They stay LISTED (the capability is real, just not from here) and are
 * disabled with the reason attached.
 *
 * The trainers route is injected because the live server predates the commit and loops never restart
 * it; the backend half is proven by tests/test_dashboard_trainer_form_support.py.
 *
 * Run: node scripts/audit-training-providers.mjs
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'
const TOKEN = fs.readFileSync(process.env.HOME + '/.strands_dashboard/local_api_token.txt', 'utf8').trim()
const b = await chromium.launch(); const ctx = await b.newContext({ viewport: { width: 1280, height: 1000 }, serviceWorkers: 'block' }); const p = await ctx.newPage()
p.on('pageerror', e => console.log('PAGEERR', e.message.slice(0, 160)))
await p.routeWebSocket('**/ws/mesh**', ws => ws.send(JSON.stringify({ type: 'snapshot', dashboard_peer_id: 'a', peers: {} })))
// The live server predates this commit (loops never restart it), so the new key is injected here;
// the backend half is proven by tests/test_dashboard_trainer_form_support.py instead.
await p.route('**/api/training/trainers**', r => r.fulfill({ status: 200, contentType: 'application/json',
  body: JSON.stringify({ trainers: ['cosmos3','fast_sac','groot','lerobot_local','mock','ppo'],
    unsupported: { ppo: 'reinforcement-learning trainers learn from a live environment, not from a recorded dataset, so they are driven from a script (RLTrainSpec) rather than this form',
                   fast_sac: 'reinforcement-learning trainers learn from a live environment, not from a recorded dataset, so they are driven from a script (RLTrainSpec) rather than this form' } }) }))
await p.route('**/api/training/datasets**', r => r.fulfill({ status: 200, contentType: 'application/json', body: '{"datasets":[]}' }))
await p.goto('http://127.0.0.1:8090/?token=' + TOKEN, { waitUntil: 'domcontentloaded' })
await p.waitForTimeout(4500)
await p.locator('button.chip:has-text("train")').first().click()
await p.waitForTimeout(3000)
const failures = []
const opts = await p.$$eval('.train-form select option', els => els.map(e => ({ t: e.textContent, d: e.disabled, title: (e.title||'').slice(0,40) })))
const hints = await p.$$eval('.train-form p.hint', els => els.map(e => e.textContent))
const byName = (n) => opts.find(o => o.t.startsWith(n))

for (const n of ['ppo', 'fast_sac']) {
  const o = byName(n)
  if (!o) { failures.push(`${n} vanished from the dropdown — the capability must stay visible`); continue }
  if (!o.d) failures.push(`${n} is still selectable, so the RLTrainSpec error is still reachable`)
  if (!/reinforcement-learning/.test(o.title)) failures.push(`${n} carries no reason`)
  if (!/not from this form/.test(o.t)) failures.push(`${n}'s label does not say why it is greyed out`)
}
// A supervised trainer must be untouched — over-refusing hides working backends.
for (const n of ['lerobot_local', 'mock', 'groot', 'cosmos3']) {
  const o = byName(n)
  if (!o) { failures.push(`${n} missing from the dropdown`); continue }
  if (o.d) failures.push(`${n} trains from a dataset but was refused`)
}
// One line, both names, the long reason printed once.
if (hints.length !== 1) failures.push(`expected one grouped explanation, got ${hints.length}`)
else {
  const h = hints[0]
  if (!/fast_sac and ppo cannot be trained from here/.test(h)) failures.push(`the explanation does not name both: ${h}`)
  if (/are a reinforcement/.test(h)) failures.push('singular/plural mismatch in the explanation')
}

await b.close()
if (failures.length) { console.error('FAIL\n' + failures.map(f => ' - ' + f).join('\n')); process.exit(1) }
console.log('training providers: ppo + fast_sac stay listed but are unselectable with their reason, the four dataset trainers are untouched, and the explanation groups both names into one sentence')
