/**
 * No audit may launch an unguarded browser.
 *
 * The guard rides along with scripts/lib/audit-browser.mjs, so importing playwright directly is how an
 * audit silently opts out of it — which is exactly what happened before the guard existed: one glob that
 * could not match '/teleop/stop' sent two real stop commands to a real arm, and nothing noticed, because
 * an escaped request looks like a fixture that did not apply.
 *
 * This check exists because THREE TIMES in this repo a rule was correct, tested, green — and never ran.
 */
import fs from 'node:fs'
import path from 'node:path'

const dir = path.dirname(new URL(import.meta.url).pathname)
const offenders = []
let checked = 0
for (const f of fs.readdirSync(dir).filter(n => n.startsWith('audit-') && n.endsWith('.mjs'))) {
  const src = fs.readFileSync(path.join(dir, f), 'utf8')
  if (!/chromium/.test(src)) continue
  checked += 1
  if (/node_modules\/playwright/.test(src)) offenders.push(f)
}
// A narrowed check must say so rather than exiting 0 on an empty sweep (this repo's own law).
if (checked === 0) { console.error('  FAIL  check-audit-guard found NO browser audits to check — it has been narrowed to nothing'); process.exit(1) }
if (offenders.length) {
  console.error(`  FAIL  ${offenders.length} of ${checked} browser audit(s) import playwright directly, so their pages have NO hardware guard:`)
  for (const o of offenders) console.error(`          ${o} — import { chromium } from './lib/audit-browser.mjs' instead`)
  process.exit(1)
}
console.log(`  PASS  every browser audit launches the guarded browser (${checked} checked)`)
