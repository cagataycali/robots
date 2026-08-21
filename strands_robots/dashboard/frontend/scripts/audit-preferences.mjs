/**
 * Q166: the four accessibility preferences, PROVEN IN A BROWSER instead of asserted in a file.
 *
 * Q162-Q165 each landed a static guard over styles.css, and static guards can all be right while the
 * app is still wrong — a rule can be shadowed, scoped inside another block, or overridden by a later
 * one, which is exactly the trap Q165 fell into (six token overrides that a later :root silently won).
 * Only a browser with the preference actually set can say what a person with that preference sees.
 *
 * So this emulates each one and reads COMPUTED styles off the real page:
 *   · reduced motion       — the busy dot must be still, not a 1000Hz strobe (iteration-count 1)
 *   · reduced transparency — the sign-in card and the modal backdrops must drop their blur
 *   · forced colours       — the three dot states must differ by something other than hue
 *   · more contrast        — the border tokens must actually be raised (the Q165 no-op class)
 * Each check names what a user would experience, because a failure here is not a style nit.
 *
 * AN AUDIT MUST PROVE ITS OWN PREMISE. Playwright 1.62 ACCEPTS `emulateMedia({ reducedTransparency })`
 * and does nothing with it — matchMedia still reports no-preference — so the first version of this file
 * reported three confident regressions in CSS that is correct. A rail that silently does nothing turns
 * an audit into a defect generator. Transparency is therefore set through CDP (which Chromium does
 * support), and every preference is CONFIRMED with matchMedia before any verdict is allowed; when the
 * preference cannot be established the audit says THAT, and never blames the page.
 */
import fs from 'node:fs'
import { chromium } from './lib/audit-browser.mjs'  // guarded browser: every page blocks unintercepted mutations

const TOKEN = fs.readFileSync(
  process.env.STRANDS_DASH_TOKEN_FILE ?? `${process.env.HOME}/.strands_dashboard/local_api_token.txt`, 'utf8').trim()
const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'

const problems = []
const notes = []
const browser = await chromium.launch()

/* A fresh context per preference: emulateMedia is per-page, and a stale page would report the
 * previous preference's computed values — a race that would invent a passing result. */
async function withPrefs({ media, cdpFeatures, query }, fn) {
  const ctx = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1280, height: 900 } })
  const page = await ctx.newPage()
  if (media) await page.emulateMedia(media)
  if (cdpFeatures) {
    const cdp = await ctx.newCDPSession(page)
    await cdp.send('Emulation.setEmulatedMedia', { features: cdpFeatures })
  }
  await page.goto(`${BASE}/?token=${TOKEN}`, { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(900)
  /* The premise: is the preference actually in effect in THIS page? */
  const inEffect = await page.evaluate(q => matchMedia(q).matches, query)
  if (!inEffect) {
    problems.push(`INCONCLUSIVE: could not put the browser into ${query} — the emulation rail did nothing, ` +
      `so this preference was NOT tested. That is a broken audit, not a broken page.`)
    await ctx.close()
    return
  }
  try { await fn(page) } finally { await ctx.close() }
}

/* The probe injects a dot of each state rather than hunting for a live robot: the preference is a
 * property of the STYLESHEET, and a fleet with no busy robot must not silently skip the check. */
const dotStyles = (page) => page.evaluate(() => {
  const out = {}
  for (const state of ['on', 'busy', 'off']) {
    const el = document.createElement('span')
    el.className = `dot ${state}`
    document.body.appendChild(el)
    const cs = getComputedStyle(el)
    out[state] = {
      background: cs.backgroundColor, borderStyle: cs.borderTopStyle, borderColor: cs.borderTopColor,
      iterations: cs.animationIterationCount, duration: cs.animationDuration,
    }
    el.remove()
  }
  return out
})

await withPrefs({ media: { reducedMotion: 'reduce' }, query: '(prefers-reduced-motion: reduce)' }, async (page) => {
  const d = await dotStyles(page)
  notes.push(`reduced motion: busy dot iterations=${d.busy.iterations} duration=${d.busy.duration}`)
  if (d.busy.iterations !== '1' && parseFloat(d.busy.duration) > 0.05)
    problems.push(`REGRESSION: with Reduce Motion set, the busy dot runs ${d.busy.iterations} iteration(s) of a ` +
      `${d.busy.duration} animation — a looping animation sped up is a STROBE, aimed at the people who asked for less motion`)
})

await withPrefs({ cdpFeatures: [{ name: 'prefers-reduced-transparency', value: 'reduce' }],
                  query: '(prefers-reduced-transparency: reduce)' }, async (page) => {
  const seen = await page.evaluate(() => {
    const probe = (cls) => {
      const el = document.createElement('div'); el.className = cls; document.body.appendChild(el)
      const cs = getComputedStyle(el); const v = cs.backdropFilter || cs.webkitBackdropFilter; el.remove(); return v
    }
    return { authcard: probe('authcard'), 'sheet-backdrop': probe('sheet-backdrop'), 'train-sheet': probe('train-sheet'),
             'drawer-backdrop': probe('drawer-backdrop'), 'detail-backdrop': probe('detail-backdrop'),
             sessionwarn: probe('sessionwarn') }
  })
  notes.push('reduced transparency: ' + Object.entries(seen).map(([k, v]) => `${k}=${v || 'none'}`).join(' '))
  for (const [name, v] of Object.entries(seen))
    if (v && v !== 'none') problems.push(`REGRESSION: with Reduce Transparency set, .${name} still blurs (${v})`)
})

await withPrefs({ media: { forcedColors: 'active' }, query: '(forced-colors: active)' }, async (page) => {
  const d = await dotStyles(page)
  notes.push(`forced colours: on=${d.on.background}/${d.on.borderStyle} busy=${d.busy.background} off=${d.off.background}/${d.off.borderStyle}`)
  const sig = (s) => `${s.background}|${s.borderStyle}`
  const uniq = new Set([sig(d.on), sig(d.busy), sig(d.off)])
  if (uniq.size < 3)
    problems.push(`REGRESSION: under forced colours only ${uniq.size} of the 3 dot states are distinguishable — ` +
      `"live", "working" and "gone" look the same, and the UA drops the glow that used to separate them`)
})

await withPrefs({ media: { contrast: 'more' }, query: '(prefers-contrast: more)' }, async (page) => {
  const t = await page.evaluate(() => {
    const cs = getComputedStyle(document.documentElement)
    return { border: cs.getPropertyValue('--border').trim(), dim: cs.getPropertyValue('--dim').trim(),
             glass1: cs.getPropertyValue('--glass-1').trim() }
  })
  notes.push(`more contrast: --border=${t.border} --dim=${t.dim} --glass-1=${t.glass1}`)
  if (t.border === '#1f2937') problems.push('REGRESSION: with More Contrast set, --border is still the 1.23:1 hairline — ' +
    'the remap is being overridden (a media :root loses to any later plain :root)')
  if (t.glass1.includes('/')) problems.push('REGRESSION: with More Contrast set, --glass-1 is still translucent — ' +
    'text sits on whatever happens to be behind it')
})

await browser.close()
for (const n of notes) console.log(`  note  ${n}`)
if (problems.length) { for (const p of problems) console.log(`  FAIL  ${p}`); process.exit(1) }
console.log('  PASS  all four accessibility preferences change the page as intended, measured in a browser')
