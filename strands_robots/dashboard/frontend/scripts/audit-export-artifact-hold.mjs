/**
 * A checkpoint the server could not confirm on disk does NOT get staged for a robot (Q36).
 *
 * The server half is unit-tested (test_dashboard_export_artifact_verdict.py) and the pure disk
 * check twice over — and that is exactly the state Q35/R5 were in when the judgment was right
 * and the screen showed nothing. The claim under test is a UI claim: "deploy holds it back, and
 * the door still opens when the operator means it", so only a browser can answer.
 *
 * Everything is INJECTED: /api/training/jobs serves one finished job, /api/training/export
 * answers with the verdict this audit wants. Nothing trains, nothing exports, nothing touches a
 * real output directory, and no robot is addressed. Safe against the live rig.
 *
 * The two directions BOTH matter here: a hold that never opens is a trap, and a hold that opens
 * silently is decoration. So this checks the box appears, that "stage it anyway" really writes
 * the deploy intent (in sessionStorage, where the run form reads it), and that a healthy export
 * stages with no interruption at all.
 *
 * Run: node scripts/audit-export-artifact-hold.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'
import fs from 'node:fs'

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []

const JOB = {
  job_id: 'audit-job-1', provider: 'mock', dataset: '/tmp/audit-dataset',
  base_model: 'lerobot/smolvla_base', output_dir: '/tmp/audit-out', steps: 1000,
  submitted_at: Date.now() / 1000 - 600,
}

const HALF_WRITTEN =
  '/tmp/audit-out/checkpoints/000100/pretrained_model has train_config.json but no weights file. '
  + 'A checkpoint directory is discovered BY ITS CONFIG, so a run killed between writing the config '
  + 'and writing model.safetensors - a crash, an OOM, a full disk, a closed lid - exports and deploys '
  + 'as if it were finished.'

const exportBody = (deployable) => deployable
  ? {
      status: 'success', deployable: true,
      text: '[mock] exported loadable artifact:\n/tmp/audit-out/checkpoints/000100/pretrained_model',
      data: { provider: 'mock', exported_model: '/tmp/audit-out/checkpoints/000100/pretrained_model' },
      artifact: { ok: true, path: '/tmp/audit-out/checkpoints/000100/pretrained_model', note: 'checked on disk only - nothing here loads the model' },
    }
  : {
      status: 'success', deployable: false,
      text: '[mock] exported loadable artifact:\n/tmp/audit-out/checkpoints/000100/pretrained_model',
      data: { provider: 'mock', exported_model: '/tmp/audit-out/checkpoints/000100/pretrained_model' },
      artifact: { ok: false, reason: 'config_without_weights', message: HALF_WRITTEN },
    }

const openTraining = async (deployable) => {
  // serviceWorkers:'block' is REQUIRED: this dashboard is a PWA and a SW-served response is not
  // interceptable by page.route, so the injected fixtures would silently never land.
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 1000 }, serviceWorkers: 'block' })
  const page = await ctx.newPage()
  const thrown = []
  page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))
  await page.route('**/api/training/jobs', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ jobs: [JOB] }) }))
  await page.route('**/api/training/status**', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'success', text: 'finished', data: { status: 'success', metrics: { latest_loss: 0.04 } } }) }))
  await page.route('**/api/training/export', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(exportBody(deployable)) }))
  // Nothing may reach a real submit even if a stray click lands.
  await page.route('**/api/training/submit', r => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'error', text: 'blocked by audit' }) }))
  await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(6000)
  await page.locator('button.chip:has-text("train")').first().click()
  // The deploy button exists only for a job whose STATUS says success, and the status arrives
  // from a 5s polling interval - not from the jobs list. Waiting a fixed 2.5s here found no
  // button and read as "the UI lost the deploy affordance"; the affordance was simply not
  // born yet. Wait for the thing, never for a duration.
  await page.locator('button:has-text("deploy")').first().waitFor({ timeout: 25000 }).catch(() => {})
  return { page, ctx, thrown }
}

const browser = await chromium.launch()

// ---- half-written checkpoint: deploy must NOT stage it
{
  const { page, ctx, thrown } = await openTraining(false)
  const deploy = page.locator('button:has-text("deploy")').first()
  if (!(await deploy.count())) {
    failures.push('no deploy button on a finished job - the audit could not reach the behaviour')
  } else {
    await deploy.click()
    await page.waitForTimeout(1500)
    const hold = page.locator('.artifact-hold').first()
    if (!(await hold.count())) {
      failures.push('a non-deployable artifact produced no hold box')
    } else {
      const text = await hold.innerText()
      if (!text.includes('BY ITS CONFIG')) failures.push('the hold box does not name the mechanism')
      if (!/stage it anyway/i.test(text)) failures.push('the hold box offers no door - a refusal with no way through is a trap')
      if (!/keep it unstaged/i.test(text)) failures.push('the hold box has no way to decline')
      if ((await hold.getAttribute('role')) !== 'alert') failures.push('the hold box is not role=alert')
    }
    // THE POINT: nothing was carried to the run form.
    const staged = await page.evaluate(() => sessionStorage.getItem('strands.deployIntent'))
    if (staged) failures.push(`a non-deployable checkpoint was staged anyway: ${staged.slice(0, 120)}`)

    // ---- the door opens, and it stages the same checkpoint with an honest provenance
    await page.locator('.artifact-hold button:has-text("stage it anyway")').first().click()
    await page.waitForTimeout(1000)
    const after = await page.evaluate(() => sessionStorage.getItem('strands.deployIntent'))
    if (!after) failures.push('"stage it anyway" did not stage anything - the override is decoration')
    else {
      const intent = JSON.parse(after)
      if (!String(intent.checkpoint).includes('pretrained_model')) failures.push('the override staged a different path than the export named')
      if (!/unconfirmed/i.test(intent.source ?? '')) failures.push('the override does not record that it overrode anything')
    }
    if (await page.locator('.artifact-hold').count()) failures.push('the hold box survived its own decision')
  }
  if (thrown.length) failures.push(`page threw: ${thrown.join(' ; ')}`)
  await ctx.close()
}

// ---- a healthy checkpoint: staged immediately, no interruption
{
  const { page, ctx, thrown } = await openTraining(true)
  await page.locator('button:has-text("deploy")').first().click()
  await page.waitForTimeout(1500)
  if (await page.locator('.artifact-hold').count()) failures.push('a healthy artifact was held back - the gate fires on good exports')
  const staged = await page.evaluate(() => sessionStorage.getItem('strands.deployIntent'))
  if (!staged) failures.push('a healthy artifact was not staged at all')
  else if (/unconfirmed/i.test(JSON.parse(staged).source ?? '')) {
    failures.push('a healthy artifact is labelled as overridden')
  }
  if (thrown.length) failures.push(`page threw (healthy): ${thrown.join(' ; ')}`)
  await ctx.close()
}

await browser.close()
if (failures.length) { console.log('FAILURES:'); for (const f of failures) console.log(`  ✗ ${f}`); process.exit(1) }
console.log('artifact hold: half-written checkpoint never reaches a robot, the override works and says so, healthy exports stage untouched')
