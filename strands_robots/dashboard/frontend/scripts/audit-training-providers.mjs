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
    unsupported: { cosmos3: "needs a training recipe TOML (extra['sft_toml']) that selects the registered experiment, and this form has no field for it - run cosmos3 from a script",
                   ppo: 'reinforcement-learning trainers learn from a live environment, not from a recorded dataset, so they are driven from a script (RLTrainSpec) rather than this form',
                   fast_sac: 'reinforcement-learning trainers learn from a live environment, not from a recorded dataset, so they are driven from a script (RLTrainSpec) rather than this form' } }) }))
// One dataset so the PLAN sentence renders (it only appears once a dataset is picked).
await p.route('**/api/training/datasets**', r => r.fulfill({ status: 200, contentType: 'application/json',
  body: JSON.stringify({ datasets: [{ repo_id: 'me/pick-cube', root: '/tmp/ds/pick-cube', label: 'pick-cube', episodes: 3 }] }) }))
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
// A trainer the form CAN drive must be untouched — over-refusing hides working backends.
for (const n of ['lerobot_local', 'mock', 'groot']) {
  const o = byName(n)
  if (!o) { failures.push(`${n} missing from the dropdown`); continue }
  if (o.d) failures.push(`${n} trains from a dataset but was refused`)
}
// Q49: cosmos3 is refused for its OWN reason (a recipe TOML), so the explanations must be two
// lines, not one merged sentence.
const rlHint = hints.find(h => /reinforcement-learning/.test(h)) || ''
if (!/fast_sac and ppo cannot be trained from here/.test(rlHint)) failures.push(`the RL explanation does not name both: ${rlHint}`)
if (/are a reinforcement/.test(rlHint)) failures.push('singular/plural mismatch in the RL explanation')
if (!hints.some(h => /cosmos3 cannot be trained from here.*recipe TOML/.test(h))) {
  failures.push('cosmos3 has no explanation of its own')
}
if (byName('cosmos3') && !byName('cosmos3').d) failures.push('cosmos3 is still selectable')

// --- Q49: groot is NOT refused; the form grows the field its validate() demands ---
// Pick the dataset first: the plan sentence is only written once one is chosen.
await p.selectOption('.train-form select >> nth=1', { index: 1 }).catch(() => {})
await p.waitForTimeout(300)
await p.selectOption('.train-form select', 'groot')
await p.waitForTimeout(400)
const gr = await p.evaluate(() => {
  const labels = [...document.querySelectorAll('.train-form label.field')]
  const emb = labels.find(l => l.querySelector('span')?.textContent === 'embodiment')
  return { present: !!emb, say: emb?.querySelector('.fieldsay')?.textContent || '',
           invalid: emb?.querySelector('input')?.getAttribute('aria-invalid'),
           story: document.querySelector('.train-story')?.textContent || '' }
})
if (!gr.present) failures.push('groot is selectable but the form still has no embodiment field')
if (!/embodiment_tag/.test(gr.say)) failures.push('the embodiment field does not quote the flag the trainer names')
if (gr.invalid !== 'true') failures.push('an empty required embodiment is not flagged')
// And the plan read-back must admit the certain refusal.
if (!/refused until you fill it in/.test(gr.story)) failures.push(`the plan sentence hides the certain refusal: ${gr.story}`)
// Switching back must remove it: an input that does nothing teaches people to ignore inputs.
await p.selectOption('.train-form select', 'lerobot_local')
await p.waitForTimeout(300)
if (await p.evaluate(() => [...document.querySelectorAll('.train-form label.field span')].some(s => s.textContent === 'embodiment'))) {
  failures.push('the embodiment field lingers on a provider that does not use it')
}

await b.close()
if (failures.length) { console.error('FAIL\n' + failures.map(f => ' - ' + f).join('\n')); process.exit(1) }
console.log('training providers: ppo/fast_sac/cosmos3 stay listed but unselectable with their own reasons, lerobot_local/mock/groot untouched, and picking groot grows the embodiment field its validate() demands (flagged empty, named in the plan sentence, gone again on another provider)')
