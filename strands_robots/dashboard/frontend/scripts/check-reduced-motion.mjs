/**
 * Q163: an infinite animation must be STILLED under prefers-reduced-motion, not sped up.
 *
 * The trap this guards is subtle and was live for months: the global escape hatch
 *   * { animation-duration: .001s !important }
 * reads like "no animation", and for a finite one it is. For `infinite alternate` it is a 1000Hz
 * strobe — the animation still runs, forever, one millisecond per cycle. Reduce Motion is set by
 * people with vestibular disorders AND by people with photosensitivity, so the rule meant to
 * protect them was producing flicker on the busy dot and the live mic ring.
 *
 * The signal that something was wrong was already in the stylesheet: four rules say
 * `animation: none` by hand inside their own reduce query. A global rule that individual rules
 * keep opting out of is not doing its job.
 *
 * So: every rule whose animation loops (`infinite`) must be stilled, either by the global cap on
 * animation-iteration-count or by naming that selector with `animation: none` in a reduce query.
 * Comments are stripped first (a prose comment naming a class fooled an earlier probe into
 * inventing selectors made of English).
 */
import fs from 'node:fs'
import path from 'node:path'

const CSS = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'src', 'styles.css')
const css = fs.readFileSync(CSS, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')

/* The global cap: a universal selector inside a reduce query that ends every loop. */
const globalCap = [...css.matchAll(/@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{([\s\S]*?)\n\}/g)]
  .some(m => /\*\s*\{[^}]*animation-iteration-count\s*:\s*1\s*!important/.test(m[1]))

/* Selectors whose animation loops, ignoring rules that only exist inside a media query. */
const withoutMedia = css.replace(/@media[^{]*\{(?:[^{}]*\{[^}]*\})*[^{}]*\}/gs, '')
const looping = new Set()
for (const [, sel, body] of withoutMedia.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
  const decl = /(?:^|[\s;])animation\s*:([^;]*)/.exec(body)
  if (!decl || !/\binfinite\b/.test(decl[1])) continue
  for (const one of sel.split(',')) {
    const s = one.trim()
    if (s && !s.startsWith('@')) looping.add(s)
  }
}

/* Named individually: `animation: none` for that selector inside a reduce query. */
const reduceQueries = [...css.matchAll(/@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{([\s\S]*?)\n?\}/g)].map(m => m[1]).join('\n')
const stilledByName = (s) => new RegExp(`${s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\{[^}]*animation\\s*:\\s*none`).test(reduceQueries)

const strobing = globalCap ? [] : [...looping].filter(s => !stilledByName(s)).sort()
console.log(`reduced motion: ${looping.size} looping animation(s), ${globalCap ? 'globally capped at one iteration' : `${looping.size - strobing.length} stilled by name`}`)
if (strobing.length) {
  console.log(`  FAIL  ${strobing.length} animation(s) become a ~1000Hz STROBE under Reduce Motion, not still:`)
  for (const s of strobing) console.log(`        ${s}`)
  console.log('  Add `animation-iteration-count: 1 !important` to the global reduce rule (preferred: it')
  console.log('  covers every future loop), or `animation: none` for each selector in a reduce query.')
  process.exit(1)
}
