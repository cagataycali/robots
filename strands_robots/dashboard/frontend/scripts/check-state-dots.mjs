/**
 * Q164: a mark whose ONLY difference between states is colour tells some people nothing.
 *
 * The fleet's status dot came in three states — on / busy / off — separated by hue and a glow.
 * That fails two different readers at once:
 *   · a screen reader gets an empty <span> (a `title` is not reliably announced, and on a touch
 *     screen there is no hover to reveal it either);
 *   · under forced colours the UA REPLACES background-color with its own palette and drops the
 *     glow, so "live", "working" and "gone" render as the same mark.
 * Both were true here: two of the three dots had no accessible name at all, and the stylesheet had
 * no forced-colors block whatsoever.
 *
 * So this requires two things that are cheap to keep and easy to lose:
 *   1. every element carrying a `dot <state>` class names its state (aria-label, or aria-hidden if
 *      the state is genuinely written beside it — silence must be a DECISION, not an oversight);
 *   2. the stylesheet distinguishes those states under `forced-colors: active` by something other
 *      than colour.
 */
import fs from 'node:fs'
import path from 'node:path'

const SRC = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'src')
const files = fs.readdirSync(SRC, { recursive: true }).map(String).filter(f => /\.tsx$/.test(f))

const unnamed = []
let dots = 0
for (const f of files) {
  const body = fs.readFileSync(path.join(SRC, f), 'utf8')
  /* Each JSX element that mentions a dot state class, taken whole so its attributes can be read. */
  for (const m of body.matchAll(/<[a-zA-Z][\s\S]*?>/g)) {
    const tag = m[0]
    if (!/\bdot (?:on|off|busy)\b/.test(tag)) continue
    if (!/className/.test(tag)) continue  // a prose mention of the class is not a rendered dot
    dots++
    if (!/aria-label=|aria-hidden=|aria-labelledby=/.test(tag)) {
      const line = body.slice(0, m.index).split('\n').length
      unnamed.push(`${f}:${line}`)
    }
  }
}

const css = fs.readFileSync(path.join(SRC, 'styles.css'), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')
const fc = /@media\s*\(forced-colors:\s*active\)\s*\{([\s\S]*?)\n\}/.exec(css)
const fcCovers = fc && ['on', 'busy', 'off'].every(st => new RegExp(`\\.dot\\.${st}\\b`).test(fc[1]))

console.log(`state dots: ${dots} rendered, ${dots - unnamed.length} naming their state; forced-colors fallback ${fcCovers ? 'covers all three states' : 'MISSING'}`)
let bad = 0
if (unnamed.length) {
  bad = 1
  console.log(`  FAIL  ${unnamed.length} status dot(s) say their state in colour only: ${unnamed.join(', ')}`)
  console.log('  Add aria-label with the same words the tooltip uses — or aria-hidden if the state is')
  console.log('  already written beside it, so the silence is a decision rather than an oversight.')
}
if (!fcCovers) {
  bad = 1
  console.log('  FAIL  no forced-colors rule distinguishes .dot.on / .dot.busy / .dot.off. Under a forced')
  console.log('  palette the UA replaces background-color and drops box-shadow: use fill AND border-style.')
}
process.exit(bad)
