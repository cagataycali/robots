/**
 * Q138: a class name in the markup that NO stylesheet defines.
 *
 * Q137's disease, generalised. There, ten rules sat in a file nobody imported, so #2486's label rows and
 * U22's death note rendered unstyled with a green build. The same silence covers a class that was simply
 * never written: JSX happily renders className="camgallery" whether or not a rule exists, so unstyled
 * markup is indistinguishable from styled markup to the compiler, the tests and the build. Only a human
 * looking at the right screen can see it — which is exactly what did not happen for days.
 *
 * This does NOT try to prove the styling is GOOD. It asks the one question that is unarguable and cheap:
 * every literal class the JSX names must appear somewhere in a stylesheet that ships.
 *
 * BASELINE: ten names were already undefined when this guard was written. They are listed — not
 * forgiven — so the guard can fail on the ELEVENTH today rather than after another arc of silent debt,
 * and so the list is a visible queue instead of a discovery. Each is real markup on a real screen
 * (a <pre> with no wrapping rule is how Q136's phone clipping happened), so removing a name from
 * KNOWN_UNDEFINED means writing the rule or deleting the class — never editing this list to be green.
 */
import fs from 'node:fs'
import path from 'node:path'

const SRC = path.resolve(import.meta.dirname, '..', 'src')
const KNOWN_UNDEFINED = new Set([
  'bubble-text',      // AgentDock: the message text inside a chat bubble
  'camgallery',       // CameraGallery: the whole gallery wrapper
  'dev-snippet',      // DevicePanel: the spawn-snippet block
  'snippet',          // DevicePanel: the <pre> inside it — UNWRAPPED, the Q136 shape
  'linklike',         // AuthGate: "use a token instead" button styled as a link
  'passkey-list',     // PasskeyList: the list wrapper
  'preset-row',       // SettingsDrawer: a row of preset buttons
  'rec-disk-notice',  // RecordPanel: the disk-space warning (rides on .train-msg, which IS defined)
  'src',              // ActivityLog: the source icon column
  'thumb-loading',    // AuthedImg: the placeholder span before a thumbnail arrives — no rule = no size
])

const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).flatMap(e =>
  e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)])
const files = walk(SRC)

const defined = new Set()
for (const f of files.filter(f => f.endsWith('.css')))
  for (const m of fs.readFileSync(f, 'utf8').matchAll(/\.([A-Za-z][\w-]*)/g)) defined.add(m[1])

const used = new Map()
for (const f of files.filter(f => f.endsWith('.tsx'))) {
  const text = fs.readFileSync(f, 'utf8')
  for (const m of text.matchAll(/className=(?:"([^"]*)"|\{`([^`]*)`\}|\{'([^']*)'\})/g)) {
    // A template's EXPRESSION is not a class list, so it is replaced by a MARKER rather than a space,
    // and every token touching that marker is dropped. Two reasons, both measured while writing this:
    // a bare space leaves the expression's own tokens ("false", a variable name) looking like missing
    // rules, and `cam-${id}` leaves the PREFIX "cam-", which is not a class either. A guard that cries
    // about `.false` and `.cam-` gets an allowlist bolted on and then stops being read.
    const chunk = (m[1] ?? m[2] ?? m[3] ?? '').replace(/\$\{[^}]*\}/g, '\u0000')
    for (const cls of chunk.split(/\s+/)) {
      if (cls.includes('\u0000')) continue
      if (/^[a-z][\w-]*$/.test(cls)) (used.get(cls) ?? used.set(cls, new Set()).get(cls)).add(path.basename(f))
    }
  }
}

const missing = [...used].filter(([c]) => !defined.has(c))
const fresh = missing.filter(([c]) => !KNOWN_UNDEFINED.has(c))
const fixed = [...KNOWN_UNDEFINED].filter(c => defined.has(c) || !used.has(c))

if (fresh.length) {
  console.error(`FAIL  ${fresh.length} class name(s) rendered by the app that NO stylesheet defines — `
    + 'unstyled markup, green build, nothing fails:')
  for (const [c, where] of fresh) console.error(`  - .${c}  (${[...where].sort().join(', ')})`)
  console.error('  Write the rule in styles.css, or drop the class from the JSX.')
  process.exit(1)
}
if (fixed.length) {
  console.error(`FAIL  ${fixed.length} name(s) in KNOWN_UNDEFINED are no longer undefined: ${fixed.join(', ')}. `
    + 'Delete them from the baseline — a stale baseline hides the next one.')
  process.exit(1)
}
console.log(`class rules: ${used.size} literal class name(s) in the JSX, ${defined.size} defined; `
  + `${missing.length} known-undefined (baseline), 0 new`)
