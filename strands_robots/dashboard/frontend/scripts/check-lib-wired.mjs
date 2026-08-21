#!/usr/bin/env node
/**
 * Does anything actually CALL each pure rule in src/lib?
 *
 * This dashboard's failure mode of the week: a rule that is correct, tested, green — and reaches no
 * screen. It has happened three times in two days, each time costing more than the feature itself:
 *   * the fleet role badge, right in lib/ and rendered from a websocket snapshot nobody had annotated;
 *   * a 404-explaining matcher written against a response shape this app never sends (dead on arrival);
 *   * a whole module (serverSkew) duplicating lib/serverAge, wired to ONE panel while the older one was
 *     already wired to every request.
 * A passing lib test says the rule is right, never that it is REACHED, so this walks the import graph
 * from the real entry points (main.tsx, App.tsx, every component) and names any lib module nothing can
 * reach. Transitively: a lib imported only by another unreachable lib is just as dead.
 *
 * Not a style rule. An unreachable module reads as a shipped feature in the diff, in PLAN.md and in a
 * commit message, so the next agent builds ON it instead of noticing that it never ran.
 *
 * Run: node scripts/check-lib-wired.mjs   (also run at the end of `npm test`)
 */
import fs from 'node:fs'
import path from 'node:path'

const SRC = path.resolve(new URL('.', import.meta.url).pathname, '../src')
const LIB = path.join(SRC, 'lib')

const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap(e =>
  e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)])

const files = walk(SRC).filter(f => /\.tsx?$/.test(f) && !/\.test\./.test(f) && !/\.d\.ts$/.test(f))
const libFiles = files.filter(f => f.startsWith(LIB + path.sep))
/** Entry points: anything that is NOT a pure rule. A component is reachable by definition — the
 *  router or a parent renders it, and an unrendered component is a different audit's problem. */
const entries = files.filter(f => !f.startsWith(LIB + path.sep))

/** Resolve a relative import from `file` to a lib module path, or null. */
const resolveLib = (file, spec) => {
  if (!spec.startsWith('.')) return null
  const base = path.resolve(path.dirname(file), spec)
  for (const cand of [base, `${base}.ts`, `${base}.tsx`, path.join(base, 'index.ts')]) {
    if (libFiles.includes(cand)) return cand
  }
  return null
}

const importsOf = (file) => {
  const body = fs.readFileSync(file, 'utf8')
  const specs = [
    ...[...body.matchAll(/from\s+'([^']+)'/g)].map(m => m[1]),
    ...[...body.matchAll(/import\(\s*'([^']+)'\s*\)/g)].map(m => m[1]),
  ]
  return specs.map(s => resolveLib(file, s)).filter(Boolean)
}

const reachable = new Set()
const queue = [...entries]
while (queue.length) {
  const f = queue.pop()
  for (const dep of importsOf(f)) {
    if (!reachable.has(dep)) { reachable.add(dep); queue.push(dep) }
  }
}

/* ---------------------------------------------------------------------------------------
 * PASS 2 (Q156): the same question one level down — does anything call each EXPORT?
 *
 * The module pass has a blind spot that cost a whole iteration to notice: it proved
 * lib/absentChildren.ts was imported, and passed happily while quietNotice() — the rule
 * that iteration existed to add — had no caller at all. A module reached by ONE screen
 * vouches for every export beside it, so a dead rule hides inside a live file.
 *
 * Three classes, deliberately not one number, because only the first is a defect:
 *   DEAD            — referenced by no other file, no test, and not even inside its own
 *                     module. Nothing on earth runs it.
 *   used-internally  — its own module uses it; `export` is merely wider than needed.
 *   test-only        — only its test imports it. INTENTIONAL here: tuning constants are
 *                     exported so a test asserts against the real number instead of a
 *                     copy of it. 71 of those, and gating on them would be wrong.
 * Type-only exports are skipped: an unused interface misleads nobody about behaviour.
 * ------------------------------------------------------------------------------------ */
const testFiles = walk(SRC).filter(f => /\.test\./.test(f))
const bodyOf = new Map([...files, ...testFiles].map(f => [f, fs.readFileSync(f, 'utf8')]))
/* Exports whose absence of a caller is KNOWN and explained. Each line is a claim the next
 * agent can check, not a mute suppression — resolve one and delete its row. */
const TOLERATED = new Map([
  ['lib/cameraState.ts :: retryDelayMs', 'documents itself as superseded by planRetry; kept "so the timing stays comparable", which nothing verifies — a parity assertion in planRetry\'s test would earn it a caller'],
  ['lib/jointHistory.ts :: heldSeconds', 'a label-honesty rule (the "60s" axis claiming more history than it holds) written for a chart that never called it — wire it to the joint history axis or drop it'],
])
const deadExports = []
let internalOnly = 0, testOnly = 0, exportCount = 0
for (const f of libFiles) {
  const body = bodyOf.get(f)
  const names = [...body.matchAll(/^export\s+(?:async\s+)?(?:function|const|class)\s+([A-Za-z_$][\w$]*)/gm)].map(m => m[1])
  for (const name of names) {
    exportCount++
    const re = new RegExp(`\\b${name.replace(/\$/g, '\\$')}\\b`, 'g')
    if (files.some(o => o !== f && re.test(bodyOf.get(o)))) continue
    if (testFiles.some(o => re.test(bodyOf.get(o)))) { testOnly++; continue }
    if ((body.match(re) || []).length > 1) { internalOnly++; continue }
    deadExports.push(`${path.relative(SRC, f)} :: ${name}`)
  }
}
const unexplained = deadExports.filter(d => !TOLERATED.has(d))
console.log(`  ${exportCount} lib value exports — ${testOnly} test-only (by design), ${internalOnly} exported wider than used, ${deadExports.length} with no caller anywhere`)
for (const d of deadExports.filter(d => TOLERATED.has(d))) console.log(`        known: ${d} — ${TOLERATED.get(d)}`)
if (unexplained.length) {
  console.log(`  FAIL  ${unexplained.length} lib export(s) nothing calls — a live module vouched for them:`)
  for (const d of unexplained) console.log(`          ${d}`)
  console.log('        → give it a caller, unexport it, or delete it. Then this guard stays honest.')
  process.exit(1)
}
const stale = [...TOLERATED.keys()].filter(k => !deadExports.includes(k))
if (stale.length) {
  console.log(`  FAIL  ${stale.length} tolerated export(s) are no longer dead — delete the stale row(s):`)
  for (const k of stale) console.log(`          ${k}`)
  process.exit(1)
}

const dead = libFiles.filter(f => !reachable.has(f)).map(f => path.relative(SRC, f)).sort()
console.log(`  ${libFiles.length} lib modules, ${reachable.size} reached from ${entries.length} entry points`)
if (!dead.length) {
  console.log('  PASS  every pure rule in lib/ is reachable from a screen')
  process.exit(0)
}
console.log(`  FAIL  ${dead.length} lib module(s) nothing can reach — tested, green, and never run:`)
for (const d of dead) console.log(`          ${d}`)
console.log('        → wire it to the screen it was written for, or delete it. A rule that reaches no')
console.log('          screen still reads as a shipped feature to the next agent.')
process.exit(1)
