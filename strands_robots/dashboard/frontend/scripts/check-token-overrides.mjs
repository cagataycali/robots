/**
 * Q165: a custom property set inside a media query is a NO-OP if a plain :root sets it later.
 *
 * Media queries add no specificity. So `@media (prefers-contrast: more) { :root { --border: … } }`
 * only wins over `:root { --border: … }` when it appears LATER in the stylesheet — and this file
 * declares its glass tokens 750 lines below the palette, which is where the trap lives. I walked
 * straight into it writing the contrast remap: the block sat at line 289, six glass tokens in it
 * were dead on arrival, and NOTHING would have said so. It compiles, it ships, the preference is
 * simply ignored — the same invisible failure as Q162/Q163, one layer down.
 *
 * This is not a style rule; it is arithmetic on line numbers, and it holds for every preference
 * block the app grows later (contrast, transparency, motion, colour scheme).
 */
import fs from 'node:fs'
import path from 'node:path'

const CSS = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'src', 'styles.css')
const css = fs.readFileSync(CSS, 'utf8').replace(/\/\*[\s\S]*?\*\//g, m => ' '.repeat(m.length))

/* Every :root block, with its position and whether it sits inside an at-rule.
 *
 * The FIRST version of this reader matched an optional `@media …{` prefix before `:root`, which moved
 * the match index back past the brace it was supposed to count — so depth read 0, every media block
 * looked top-level, and the guard printed "0 media :root block(s), 0 dead override(s)" while staring
 * at one. A guard that cannot see its subject reports a FALSE GREEN, which is worse than no guard, so
 * the count is printed above and this file refuses to be silent about zero. */
const depthAt = (upto) => {
  let d = 0
  for (const ch of css.slice(0, upto)) { if (ch === '{') d++; else if (ch === '}') d-- }
  return d
}
/* Braces must balance, and this is not pedantry: CSS parsers RECOVER silently from an imbalance,
 * so a missing `}` does not error — it SCOPES every following rule inside the last open block. I did
 * exactly this while writing Q165's remap (an orphan `}` left behind by a move, and my own block left
 * unclosed), which quietly put ~150 rules inside `prefers-contrast: more`. esbuild bundled it without
 * a word and the dist shipped. Depth also has to be right for the check below to mean anything. */
{
  let d = 0, firstNegative = null
  for (let i = 0; i < css.length; i++) {
    if (css[i] === '{') d++
    else if (css[i] === '}' && --d < 0 && firstNegative === null) firstNegative = css.slice(0, i).split('\n').length
  }
  if (d !== 0 || firstNegative !== null) {

console.log(`token overrides: FAIL — styles.css braces do not balance (final depth ${d}` +
      (firstNegative ? `, first stray closer at line ${firstNegative}` : '') + ').')
    console.log('  A CSS parser recovers silently from this: the rules AFTER the break end up scoped inside')
    console.log('  whatever block is still open, which is how a whole screen ends up behind a media query.')
    process.exit(1)
  }
}

const roots = []
for (const m of css.matchAll(/:root\s*\{([^}]*)\}/g)) {
  roots.push({
    at: m.index,
    body: m[1],
    inMedia: depthAt(m.index) > 0,
    line: css.slice(0, m.index).split('\n').length,
  })
}
if (roots.length === 0) {
  console.log('token overrides: FAIL — no :root block found at all, so this guard proved nothing')
  process.exit(1)
}

const dead = []
for (const r of roots.filter(r => r.inMedia)) {
  for (const [, tok] of r.body.matchAll(/(--[a-z0-9-]+)\s*:/g)) {
    const shadowing = roots.find(o => !o.inMedia && o.at > r.at && new RegExp(`(^|;)\\s*${tok}\\s*:`).test(o.body))
    if (shadowing) dead.push(`${tok} (media :root at line ${r.line}, plain :root wins at line ${shadowing.line})`)
  }
}

/* Q166 GENERALISES Q165: it is not only custom properties. ANY declaration inside a preference block
 * loses to a plain rule with the same selector later in the file. That is how the sign-in card kept its
 * blur: the reduced-transparency block sat at line 1173 and .authcard was declared at 1336, so five of
 * six surfaces obeyed the preference and one did not — a block that is 80% effective looks entirely
 * correct in review, and only a browser with the preference set can tell you otherwise.
 *
 * Preference blocks therefore belong at the END of the stylesheet, and this proves it rather than
 * asking anyone to remember. */
const PREF = /@media[^{]*(prefers-|forced-colors)/
const spans = []
{
  let d = 0
  for (let i = 0; i < css.length; i++) {
if (css[i] === '@' && d === 0) {
  const head = css.slice(i, css.indexOf('{', i) + 1)
  if (PREF.test(head)) {
    let k = css.indexOf('{', i), dd = 0
    for (; k < css.length; k++) {
      if (css[k] === '{') dd++
      else if (css[k] === '}' && --dd === 0) break
    }
    spans.push({ start: i, end: k, line: css.slice(0, i).split('\n').length })
  }
}
if (css[i] === '{') d++
else if (css[i] === '}') d--
  }
}

const inSpan = (at) => spans.some(s => at > s.start && at < s.end)
const shadowed = []
for (const span of spans) {
  const body = css.slice(css.indexOf('{', span.start) + 1, span.end)
  for (const [, sel, decls] of body.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
const selector = sel.trim()
if (!selector || selector === ':root') continue           // custom properties handled above
const props = [...decls.matchAll(/(^|;)\s*([a-z-]+)\s*:/g)].map(m => m[2])
for (const m of css.matchAll(new RegExp(`(^|\\})\\s*${selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\{([^}]*)\\}`, 'g'))) {
  const at = m.index
  if (at <= span.end || inSpan(at)) continue
  const hit = props.filter(pr => new RegExp(`(^|;)\\s*${pr}\\s*:`).test(m[2]))
  if (hit.length) shadowed.push(`${selector} { ${hit.join(', ')} } — preference block at line ${span.line}, ` +
    `plain rule wins at line ${css.slice(0, at).split('\n').length}`)
}
  }
}
if (shadowed.length) dead.push(...shadowed)

console.log(`token overrides: ${roots.filter(r => r.inMedia).length} media :root block(s), ${spans.length} preference block(s), ${dead.length} dead override(s)`)
if (dead.length) {
  console.log('  FAIL  these preference overrides are silently ignored — a media query adds no specificity,')
  console.log('        so a plain :root or rule LATER in the file wins and the preference does nothing:')
  for (const d of dead) console.log(`        ${d}`)
  console.log('  Move the media block BELOW the last plain declaration it means to override — preference')
  console.log('  blocks belong at the END of the stylesheet, which is where this file now keeps them.')
  process.exit(1)
}
