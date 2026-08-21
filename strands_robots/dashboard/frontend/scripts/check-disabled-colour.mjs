/**
 * Q147 — A CLICKABLE THING MAY NOT WEAR THE DISABLED COLOUR.
 *
 * styles.css declares --dim to be the disabled colour (.btn:disabled, .btn.go:disabled, .btn.ghost:disabled
 * all use it). Five clickable rules wore it anyway: the nav chips, the settings tabs, the dock's minimise
 * button, the apply chip and the glass button. audit-primary-actions had been reporting the consequence for
 * days without failing it — "⏺ record" flagged on every screen as an enabled action that looks disabled —
 * because it cannot tell nav from a primary action. This can: the palette's own law is checkable.
 *
 * Contrast was never the issue (--dim on --panel2 is 5.77:1, AA-clear for normal text). The defect is a
 * MEANING collision: a palette that says "this colour means you cannot click it" and then paints the entry
 * point to the record screen with it teaches the operator to distrust the signal everywhere else.
 *
 * Static text may use --dim freely — a table header is not promising to be clickable. The rule applies to
 * selectors that name an interactive component, minus those qualified by :disabled, which is the one place
 * the colour is exactly right.
 *
 * Run: node scripts/check-disabled-colour.mjs
 */
import fs from 'node:fs'
import path from 'node:path'

const css = fs.readFileSync(path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'src',
  'styles.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, m => m.replace(/[^\n]/g, ' ')) // comments lie
const INTERACTIVE = /(^|[\s,.#])(chip|tab|btn|gbtn|dock-min|dock-send|ep-box|link)\b/
const offenders = []
for (const m of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
  const sel = m[1].trim().replace(/\s+/g, ' ')
  if (!INTERACTIVE.test(sel) || /:disabled|\[disabled\]|::placeholder/.test(sel)) continue
  if (/(^|[^-\w])color: *var\(--dim\)/.test(m[2]))
    offenders.push(`${css.slice(0, m.index).split('\n').length}: ${sel.slice(0, 70)}`)
}
if (offenders.length) {
  console.error(`FAIL  ${offenders.length} interactive rule(s) painted with --dim, which this stylesheet `
    + 'reserves for DISABLED — an enabled control that looks disabled does not get clicked:')
  for (const o of offenders) console.error(`  - ${o}`)
  console.error('  Use --quiet for live-but-restrained controls; --dim only under :disabled.')
  process.exit(1)
}
console.log('disabled colour: no interactive rule wears --dim (the disabled token)')
