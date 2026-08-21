/**
 * Q162: every glass surface must have a fallback when the OS asks for less transparency.
 *
 * `prefers-reduced-transparency` is not a taste setting — it is forwarded by macOS/iOS
 * "Reduce transparency", which people turn on for vestibular reasons and for plain legibility.
 * Honouring it is a one-line rule per surface, and the app already honoured it for eight.
 *
 * The trouble is HOW that list is kept: eight components hand-roll the glass look (background +
 * border + backdrop-filter) rather than wearing the unadopted `.glass` kit, so the fallback list
 * is maintained by hand and drifts silently — nothing renders differently for the developer, who
 * almost certainly does not have the preference set. It had drifted to exactly the worst places:
 * the SIGN-IN CARD (the first thing such a person ever sees here), the training sheet, the session
 * warning, the camera-state chip and all three modal backdrops.
 *
 * So this asks the stylesheet the whole question instead of trusting the list: which selectors
 * apply a backdrop-filter, and does each one appear inside the reduce block? Comments are stripped
 * FIRST — a prose comment mentioning a class name fooled the throwaway probe that found this and
 * produced eight phantom "selectors" made of English.
 *
 * Backdrops are allowed to keep their dim (removing it would leave a sheet floating on live
 * content); they only have to drop the blur, which is what this checks.
 */
import fs from 'node:fs'
import path from 'node:path'

const CSS = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'src', 'styles.css')
const css = fs.readFileSync(CSS, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')

/* The reduce block, taken as a whole (nested rules and all) so its selectors can be read. */
const start = css.indexOf('@media (prefers-reduced-transparency: reduce)')
if (start < 0) {
  console.log('reduced transparency: FAIL — no @media (prefers-reduced-transparency: reduce) block at all')
  process.exit(1)
}
let depth = 0, end = start
for (let i = css.indexOf('{', start); i < css.length; i++) {
  if (css[i] === '{') depth++
  else if (css[i] === '}' && --depth === 0) { end = i; break }
}
const reduceBlock = css.slice(start, end)

/* Selectors that apply the glass effect, ignoring anything already inside a media query
 * (a rule that only exists under a query is that query's business). */
const withoutMedia = css.replace(/@media[^{]*\{(?:[^{}]*\{[^}]*\})*[^{}]*\}/gs, '')
const surfaces = new Set()
for (const [, sel, body] of withoutMedia.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
  if (!/(^|[^-])backdrop-filter\s*:/.test(body)) continue
  if (/backdrop-filter\s*:\s*none/.test(body)) continue
  for (const one of sel.split(',')) {
    const s = one.trim()
    if (s && !s.startsWith('@')) surfaces.add(s)
  }
}

/* A surface is covered when the reduce block names it, or names the element it modifies —
 * `.glass.lifted` is answered by `.glass`, and a pseudo-class variant by its base. */
const base = (s) => s.split(':')[0].trim()
const covered = (s) =>
  reduceBlock.includes(s) || reduceBlock.includes(base(s)) ||
  base(s).split('.').filter(Boolean).some(cls => reduceBlock.includes(`.${cls} `) || reduceBlock.includes(`.${cls},`) || reduceBlock.includes(`.${cls}{`))

const missing = [...surfaces].filter(s => !covered(s)).sort()
console.log(`reduced transparency: ${surfaces.size} glass surface(s), ${surfaces.size - missing.length} with a fallback`)
if (missing.length) {
  console.log(`  FAIL  ${missing.length} surface(s) still blur when the OS asks for less transparency:`)
  for (const m of missing) console.log(`        ${m}`)
  console.log('  Add each to @media (prefers-reduced-transparency: reduce): backdrop-filter: none, plus an')
  console.log('  opaque background if the rule relies on the blur to be readable.')
  process.exit(1)
}
