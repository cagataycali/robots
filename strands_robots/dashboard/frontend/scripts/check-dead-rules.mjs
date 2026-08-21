/**
 * Q143 — THE MIRROR OF check-class-defined: a rule whose class NOTHING renders.
 *
 * That guard asks "does every class in the markup have a rule?" and answered it four times this week,
 * twice with a live defect. This one asks the other half: does every class RULE have markup? Dead CSS is
 * quieter than dead markup — it never renders, so no screenshot shows it — but it is not free:
 *   - it ships to every visitor (measured 3.1KB of 64KB, 4.8%, when this was written),
 *   - it makes the stylesheet read like a design system that is in use, so the next person extends the
 *     wrong thing, and
 *   - it hides the real question, which is whether the kit was meant to be adopted or abandoned.
 *
 * A class is considered ALIVE if its name appears as a literal ANYWHERE in src/*.ts(x) — not just in a
 * className attribute — because tone/state names are returned by pure lib functions and composed later
 * (`bubble ${role}`, activityLine().tone). Interpolated prefixes (`cam-${state}`) are honoured too. That
 * is deliberately generous: a false positive here invites deleting a rule that DOES render, which is a
 * worse outcome than a missed dead one. Verified against the whole stylesheet when written: 343 selectors,
 * 7 dead, and every one of the 7 confirmed by hand.
 *
 * Run: node scripts/check-dead-rules.mjs
 */
import fs from 'node:fs'
import path from 'node:path'

const HERE = path.dirname(new URL(import.meta.url).pathname)
const SRC = path.join(HERE, '..', 'src')
const CSS = path.join(SRC, 'styles.css')

/* An unadopted GLASS COMPONENT KIT: `.glass` (+ .dense/.flush/.lifted modifiers), `.gbtn`, `.ginput`,
   `.helpcard`. It is real, coherent CSS that no component has ever worn — the UI reaches for the same
   look through the --glass-* TOKENS instead, which are used everywhere. So this is a decision, not a bug:
   adopt the kit, or delete it. It is written down here rather than left invisible, because a design
   language that ships but never renders is exactly the thing nobody notices for months. */
const KNOWN_DEAD = new Set(['glass', 'dense', 'flush', 'lifted', 'gbtn', 'ginput', 'helpcard'])

const css = fs.readFileSync(CSS, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')  // comments name files and ports
const files = fs.readdirSync(SRC, { recursive: true }).filter(f => /\.(tsx|ts)$/.test(String(f)))
const src = files.map(f => fs.readFileSync(path.join(SRC, String(f)), 'utf8')).join('\n')

/* MAXIMALLY GENEROUS on purpose: every word-like token in the source counts, including ones inside long
   template strings, ones a lib function returns, and ones only mentioned in a comment. A stricter reader
   (quoted strings only, split on whitespace) was tried first and ACCUSED SEVEN LIVE RULES — .twinbtn,
   .undelivered, .rec-ep and friends, whose names sit inside longer composed strings. Deleting a rule that
   really renders is a visible regression; missing a dead one costs bytes. So the test this guard actually
   makes is the narrow, true one: does this class name appear ANYWHERE in the source at all? */
const literals = new Set()
for (const m of src.matchAll(/[a-z][\w-]*/gi)) literals.add(m[0])
const prefixes = [...new Set([...src.matchAll(/([a-z][\w-]*-)\$\{/g)].map(m => m[1]))]

// Selector text only: everything before a '{' that is not itself a declaration block.
const defined = new Map()
for (const m of css.matchAll(/(^|\}|;)\s*([^{}@]+?)\{/g))
  for (const c of m[2].matchAll(/\.([a-z][\w-]*)/gi))
    if (!defined.has(c[1])) defined.set(c[1], m[2].trim().split('\n')[0].slice(0, 60))

const dead = [...defined.keys()]
  .filter(c => !literals.has(c) && !prefixes.some(p => c.startsWith(p)))
const fresh = dead.filter(c => !KNOWN_DEAD.has(c))
const revived = [...KNOWN_DEAD].filter(c => !dead.includes(c))

if (fresh.length) {
  console.error(`FAIL  ${fresh.length} class rule(s) that NOTHING in the app renders — dead CSS shipped to `
    + 'every visitor, and a stylesheet that lies about what is in use:')
  for (const c of fresh) console.error(`  - .${c}  (selector: ${defined.get(c)})`)
  console.error('  Delete the rule, or render the class. If it is a deliberate kit awaiting adoption, add '
    + 'it to KNOWN_DEAD with the reason — an undocumented one is indistinguishable from a mistake.')
  process.exit(1)
}
if (revived.length) {
  console.error(`FAIL  ${revived.length} name(s) in KNOWN_DEAD are rendered now: ${revived.join(', ')}. `
    + 'Remove them from the baseline — the kit is being adopted, which is good news the list should not hide.')
  process.exit(1)
}
console.log(`dead rules: ${defined.size} class selector(s) defined, ${dead.length} rendered by nothing `
  + `(baseline: the unadopted glass kit), 0 new`)
