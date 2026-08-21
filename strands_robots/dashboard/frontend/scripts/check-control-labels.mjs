/**
 * Q144 — every form control must have an ACCESSIBLE NAME, and a placeholder is not one.
 *
 * A placeholder disappears the moment you type. On the camera row that means five filled-in numbers with
 * nothing saying which is fps and which is height; to a screen reader it means "edit text", full stop; and
 * on the e-stop sheet it meant an unnamed password box next to a lockout. None of the existing audits
 * caught this: touch-targets measures SIZE, keyboard-path measures REACHABILITY, and a control can be big,
 * tabbable and still anonymous.
 *
 * A control counts as named if it sits inside a <label>, carries aria-label / aria-labelledby, or has an
 * id paired with a label's htmlFor. `.field` labels (the dominant pattern here) wrap their input, so most
 * of the 75 controls pass for free.
 *
 * PARSING LESSON, paid for twice in this repo (Q143 was the CSS half): read attributes with a real
 * scanner. A lazy /<input[\s\S]*?>/ ends the tag at the FIRST '>' — which in JSX is often inside an
 * expression, e.g. placeholder={n > 0 ? … } — and the attributes after it become invisible. That falsely
 * accused TrainingTab's episode box, which has carried an aria-label all along. This walks forward from
 * the tag name tracking brace depth and quotes, so the tag ends where JSX says it ends.
 *
 * SECOND SECTION (Q148) — buttons whose entire label is a bare GLYPH: <button>✕</button>. Q146 established
 * that a button's name usually cannot be judged from source, because it is a dynamic expression, and homed
 * that check in a browser audit. But that audit walks the six nav screens, so a control living only inside a
 * MODAL is outside it — and a close button is exactly that shape. This covers the statically CERTAIN subset:
 * inner content with no expression braces, no JSX child and no ASCII letters cannot be anything but a glyph,
 * wherever it renders. Measured 0 unnamed (all 147 already carry aria-label or title), so it is fatal on
 * arrival rather than carrying a baseline. The two checks are complements, not duplicates: source sees every
 * modal, the browser sees every dynamic label, and neither alone would have been enough.
 *
 * Run: node scripts/check-control-labels.mjs
 */
import fs from 'node:fs'
import path from 'node:path'

const SRC = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'src')
const files = fs.readdirSync(SRC, { recursive: true }).filter(f => /\.tsx$/.test(String(f)))

/** attributes of a JSX tag opened at `start`, brace- and quote-aware. */
function tagAttrs(src, start) {
  let i = start, depth = 0, quote = null
  while (i < src.length) {
    const c = src[i]
    if (quote) { if (c === quote) quote = null }
    else if (c === '"' || c === "'" || c === '`') quote = c
    else if (c === '{') depth++
    else if (c === '}') depth--
    else if (c === '>' && depth === 0) return src.slice(start, i)
    i++
  }
  return src.slice(start)
}

/** index of the '>' that closes the tag opened at `start` (same brace/quote rules as tagAttrs). */
function tagEnd(src, start) {
  let i = start, depth = 0, quote = null
  while (i < src.length) {
    const c = src[i]
    if (quote) { if (c === quote) quote = null }
    else if (c === '"' || c === "'" || c === '`') quote = c
    else if (c === '{') depth++
    else if (c === '}') depth--
    else if (c === '>' && depth === 0) return i
    i++
  }
  return src.length
}

let total = 0
const anonymous = []
for (const f of files) {
  // COMMENTS LIE — the same trap as Q143's stylesheet scan: DevicePanel's Q77 note says "The <select>
  // is only correct at render time", and a naive scan reported that prose as an unnamed control. Block
  // comments cover JSX's {/* … */} too. Offsets are preserved (blanked, not removed) so the reported
  // line numbers still point at real code.
  const raw = fs.readFileSync(path.join(SRC, String(f)), 'utf8')
  const src = raw.replace(/\/\*[\s\S]*?\*\//g, m => m.replace(/[^\n]/g, ' '))
  const htmlFor = new Set([...src.matchAll(/htmlFor=["{]?["']?([\w-]+)/g)].map(m => m[1]))
  for (const m of src.matchAll(/<(input|select|textarea)[\s/>]/g)) {
    const attrs = tagAttrs(src, m.index + 1 + m[1].length)
    if (/type=["']hidden["']/.test(attrs)) continue
    total++
    const before = src.slice(0, m.index)
    const wrapped = before.lastIndexOf('<label') > before.lastIndexOf('</label>')
    const aria = /aria-label(?:ledby)?[=\s]/.test(attrs)
    const idm = attrs.match(/\bid=["']([\w-]+)["']/)
    if (wrapped || aria || (idm && htmlFor.has(idm[1]))) continue
    anonymous.push({ where: `${f}:${before.split('\n').length}`, tag: m[1],
      placeholder: (attrs.match(/placeholder=["']([^"']+)/) ?? [])[1] ?? '' })
  }
}

/* Q148: a button labelled only by a glyph, in any file — including the sheets no page audit opens. */
let buttons = 0
const glyphOnly = []
for (const f of files) {
  const raw = fs.readFileSync(path.join(SRC, String(f)), 'utf8')
  const src = raw.replace(/\/\*[\s\S]*?\*\//g, m => m.replace(/[^\n]/g, ' '))
  for (const m of src.matchAll(/<button[\s/>]/g)) {
    const end = tagEnd(src, m.index + 7)
    const close = src.indexOf('</button>', end)
    if (close < 0) continue
    buttons++
    const attrs = src.slice(m.index + 7, end)
    const inner = src.slice(end + 1, close).trim()
    const named = /aria-label(?:ledby)?[=\s]/.test(attrs) || /\btitle=/.test(attrs)
    if (!named && inner.length && !/[{<]/.test(inner) && !/[A-Za-z]/.test(inner))
      glyphOnly.push({ where: `${f}:${src.slice(0, m.index).split('\n').length}`, inner })
  }
}
if (glyphOnly.length) {
  console.error(`FAIL  ${glyphOnly.length} button(s) whose whole label is a glyph, with no aria-label or `
    + 'title — the operator has to click it to learn what it does, and a screen reader reads the glyph:')
  for (const g of glyphOnly) console.error(`  - ${g.where}  <button>${g.inner}</button>`)
  process.exit(1)
}

if (anonymous.length) {
  console.error(`FAIL  ${anonymous.length} of ${total} form control(s) have NO accessible name — a `
    + 'placeholder vanishes as soon as the operator types, and a screen reader reads them as "edit text":')
  for (const a of anonymous) console.error(`  - <${a.tag}> ${a.where}`
    + (a.placeholder ? `  (placeholder only: "${a.placeholder}")` : '  (nothing at all)'))
  console.error('  Wrap it in a <label>, or give it aria-label — whichever suits the layout.')
  process.exit(1)
}
console.log(`control labels: ${total} form control(s) named · ${buttons} button(s), none labelled by a `
  + 'bare glyph alone')
