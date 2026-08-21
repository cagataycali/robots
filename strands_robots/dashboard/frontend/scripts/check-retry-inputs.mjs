#!/usr/bin/env node
/**
 * Does every socket that retries pass the WHOLE truth to planRetry?
 *
 * planRetry decides whether to knock again. Each piece of evidence it reads is OPTIONAL in the type,
 * and that is correct for the module — a pure test constructs the one field it is about. It is wrong
 * for a real call site: an omitted field is not "unknown", it silently means "no", and the plan comes
 * back saying retry when the honest answer is stop.
 *
 * Paid for twice in two days, both times the same shape:
 *   * Q88 added `sessionExpired` after 19.3 hours of refused reopens from one phone;
 *   * Q102 added `pageRefused` (a token INVALID rather than expired: rotated by a restart, a stale
 *     ?token= link, a revoked session) — and landed it wired to the camera tile only, so the fleet
 *     socket kept reconnecting against a door that had already said no. A whole extra iteration.
 *
 * TypeScript cannot catch this: every field is legitimately optional. So the rule lives here, where a
 * new call site (or a new piece of evidence) fails loudly instead of quietly under-informing the plan.
 *
 * Run: node scripts/check-retry-inputs.mjs   (also run at the end of `npm test`)
 */
import fs from 'node:fs'
import path from 'node:path'

const SRC = path.resolve(new URL('.', import.meta.url).pathname, '../src')

/**
 * Evidence a LIVE socket must always hand over. Not "every field of SocketOutcome": `code` is absent
 * when a handshake never produced one, and `recentOpens` belongs to tiles that keep an open log. These
 * two are different — they describe the PAGE, are available everywhere at zero cost, and their absence
 * always reads as "fine, retry".
 */
const REQUIRED = ['sessionExpired', 'pageRefused']

const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap(e =>
  e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)])

const files = walk(SRC).filter(f => /\.tsx?$/.test(f) && !/\.test\./.test(f) && !/\.d\.ts$/.test(f))

/** The argument text of a call, by matching brackets — regexes cannot see nesting. */
function callArg(src, from) {
  let depth = 0
  for (let i = from; i < src.length; i++) {
    const c = src[i]
    if (c === '(' || c === '{') depth++
    else if (c === ')' || c === '}') {
      depth--
      if (depth === 0) return src.slice(from, i + 1)
    }
  }
  return src.slice(from)
}

const problems = []
let sites = 0

for (const file of files) {
  const src = fs.readFileSync(file, 'utf8')
  // The definition and its own re-exports are not call sites.
  if (/export function planRetry/.test(src)) continue
  const re = /planRetry\s*\(/g
  let m
  while ((m = re.exec(src)) !== null) {
    sites++
    const arg = callArg(src, m.index + m[0].length - 1)
    const line = src.slice(0, m.index).split('\n').length
    const missing = REQUIRED.filter(k => !new RegExp(`(^|[^\\w.])${k}\\s*[:,}]`).test(arg))
    if (missing.length) {
      problems.push(`${path.relative(SRC, file)}:${line} planRetry() omits ${missing.join(', ')}`)
    }
  }
}

// A guard that can be NARROWED must say what it looked at, and must not pass while empty.
console.log(`  ${sites} planRetry call site(s) checked in ${files.length} files`)
if (sites === 0) {
  console.log('  FAIL  no planRetry call site found — either the retry rule is unwired, or this guard is')
  console.log('        looking in the wrong place. Both are worse than the bug it guards against.')
  process.exit(1)
}
if (problems.length) {
  console.log('  FAIL  a retrying socket is deciding on partial evidence:')
  for (const p of problems) console.log(`        ${p}`)
  console.log(`        Each omitted field reads as "no" — pass ${REQUIRED.join(' and ')}, or the socket`)
  console.log('        will keep knocking on a door that already refused it.')
  process.exit(1)
}
console.log(`  PASS  every retrying socket passes ${REQUIRED.join(' + ')}`)
