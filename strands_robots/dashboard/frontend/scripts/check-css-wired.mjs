/**
 * Q137: every stylesheet in src/ must be REACHABLE from the entry point.
 *
 * The sibling of check-lib-wired.mjs, and it exists because the same failure happened three times in
 * this one file: src/index.css was imported by NOTHING, so #2486's episode-labels disclosure and U22's
 * death note shipped with class names no stylesheet defined — and then a Q136 phone fix was appended to
 * the same dead file. Every one of those builds was green. A stylesheet is not code the compiler checks:
 * an unreferenced .css file is silently omitted from the bundle, and unstyled markup still renders, so
 * nothing anywhere fails. The only witness is a rule that never takes effect on screen.
 *
 * Reachability is transitive: main.tsx (or any .tsx/.ts) may import a stylesheet, and a stylesheet may
 * @import another.
 */
import fs from 'node:fs'
import path from 'node:path'

const SRC = path.resolve(import.meta.dirname, '..', 'src')
const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap(e =>
  e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)])

const files = walk(SRC)
const sheets = files.filter(f => f.endsWith('.css'))
const importers = files.filter(f => /\.(tsx|ts|jsx|js|mjs|css)$/.test(f))

// Which sheets does any importer name? A basename match is deliberately generous: this asks the weaker,
// unarguable question ("is it mentioned at all"), so a passing result can never be a false green.
const mentioned = new Set()
for (const f of importers) {
  const src = fs.readFileSync(f, 'utf8')
  for (const s of sheets) {
    if (f === s) continue
    if (src.includes(path.basename(s))) mentioned.add(s)
  }
}

const dead = sheets.filter(s => !mentioned.has(s))
const rel = s => path.relative(path.resolve(SRC, '..'), s)

if (dead.length) {
  console.error(`FAIL  ${dead.length} of ${sheets.length} stylesheet(s) in src/ are imported by nothing — `
    + 'their rules are omitted from the bundle and the markup renders unstyled, with a green build:')
  for (const s of dead) {
    const rules = (fs.readFileSync(s, 'utf8').match(/^\s*[.#@][^{]*\{/gm) ?? []).length
    console.error(`  - ${rel(s)} (${rules} rule(s) that never take effect)`)
  }
  console.error('  Either import it from main.tsx or move its rules into a sheet that is imported.')
  process.exit(1)
}
console.log(`css wired: ${sheets.length} stylesheet(s) in src/, every one reachable from an import `
  + `(${sheets.map(rel).join(', ')})`)
