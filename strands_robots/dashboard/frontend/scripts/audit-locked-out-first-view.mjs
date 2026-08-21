/**
 * Q142: what a locked-out operator meets, with NO token at all.
 *
 * Every other audit in here arrives holding `?token=…`, so the sealed door itself — the first and only
 * screen a remote operator sees when their passkey misbehaves — was the least-examined page in the
 * dashboard. This one opens the app with NO credential of any kind (a fresh context, so nothing is in
 * localStorage) and reads the hierarchy.
 *
 * The claim: there is exactly ONE loud action (the passkey), and the token fallback is OFFERED, not
 * urged. Measured before the rule existed, `.btn linklike` had no rule anywhere and inherited the whole
 * .btn look — 276x50, solid panel background, 15px, full text colour — so the fallback shouted as loudly
 * as the recommended path on the one screen where an operator has to choose between them, and the
 * ceremony that hangs (the Aug-19 iOS wedge) is exactly when they are reading it most carefully.
 *
 * It also guards the opposite overcorrection: quiet must not mean unhittable. A link-looking control on a
 * phone is still something a thumb has to find, so the tap target stays >= 24px and the keyboard ring
 * stays reachable.
 *
 * Run: node scripts/audit-locked-out-first-view.mjs   (running dashboard on :8090 + node playwright)
 */
import { chromium } from '/Users/cagatay/.tiny/npm/node_modules/playwright/index.mjs'

const BASE = process.env.STRANDS_DASH_URL ?? 'http://127.0.0.1:8090'
const failures = []
const browser = await chromium.launch()
// Phone width, and serviceWorkers blocked so a cached shell cannot answer for the live one.
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, serviceWorkers: 'block' })
const page = await ctx.newPage()
const thrown = []
page.on('pageerror', e => thrown.push(String(e.message).slice(0, 160)))

await page.goto(BASE, { waitUntil: 'domcontentloaded' })   // NO ?token, NO stored credential
await page.waitForTimeout(3000)

const gate = page.locator('.authgate').first()
if (!(await gate.count())) {
  failures.push('no auth gate with no credential at all — either this server is unsealed (check it, that '
    + 'is a security fact, not a UI one) or the gate failed to render for someone who cannot get past it')
} else {
  const text = (await gate.innerText()).replace(/\s+/g, ' ')
  if (!/passkey/i.test(text)) failures.push(`the gate never says what unlocks it: "${text.slice(0, 120)}"`)

  const shape = await page.evaluate(() => {
    const g = document.querySelector('.authgate')
    const box = e => { const b = e.getBoundingClientRect(); return { w: Math.round(b.width), h: Math.round(b.height) } }
    const loud = [...g.querySelectorAll('button.btn.go')].map(e => ({ text: e.innerText.trim(), ...box(e) }))
    const link = g.querySelector('button.linklike')
    const cs = link ? getComputedStyle(link) : null
    const primary = g.querySelector('button.btn.go')
    const ps = primary ? getComputedStyle(primary) : null
    return {
      loud, hasLink: !!link,
      link: link ? { ...box(link), bg: cs.backgroundColor, font: parseFloat(cs.fontSize),
        decoration: cs.textDecorationLine, opaque: !/rgba\(0, 0, 0, 0\)|transparent/.test(cs.backgroundColor) } : null,
      primaryFont: ps ? parseFloat(ps.fontSize) : null,
    }
  })

  if (shape.loud.length !== 1) {
    failures.push(`${shape.loud.length} loud (.btn.go) action(s) on the sealed door — `
      + `${shape.loud.map(b => `"${b.text}"`).join(', ') || 'none'}. Exactly one path should be urged, or `
      + 'an operator whose passkey just hung cannot tell which door is the real one')
  }
  if (!shape.hasLink) {
    failures.push('no token fallback offered on the gate — when the passkey ceremony wedges (it has, on '
      + 'iOS) this screen is a dead end with no second door')
  } else {
    const l = shape.link
    console.log(`  gate: ${shape.loud.length} loud action ("${shape.loud[0]?.text ?? '—'}"), fallback `
      + `${l.w}x${l.h} ${l.font}px ${l.decoration} bg=${l.bg}`)
    if (l.opaque || l.font >= shape.primaryFont) failures.push(`the token fallback is as loud as the `
      + `primary (bg ${l.bg}, ${l.font}px vs the primary's ${shape.primaryFont}px) — two equal buttons `
      + 'make the recommended path ambiguous exactly when the operator is already stuck')
    if (l.h < 24) failures.push(`the fallback is only ${l.h}px tall — quiet must not mean unhittable; a `
      + 'link-looking control on a phone is still a thumb target (24px floor)')
  }

  // The keyboard must reach it: a button that stops looking like a button is still the only second door.
  const reached = await page.evaluate(() => {
    const link = document.querySelector('.authgate button.linklike')
    if (!link) return null
    link.focus()
    return document.activeElement === link
  })
  if (reached === false) failures.push('the token fallback cannot take keyboard focus — unreachable without a mouse')
}
if (thrown.length) failures.push(`page errors on the gate: ${thrown.join(' | ')}`)

await ctx.close(); await browser.close()
if (failures.length) { for (const f of failures) console.error(`  FAIL  ${f}`); process.exit(1) }
console.log('locked-out first view: one loud path (the passkey), a quieter but hittable token fallback, no page errors')
