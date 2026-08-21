/**
 * Q145 — THE ANSWERS AN OPERATOR WAITS FOR MUST ANNOUNCE THEMSELVES.
 *
 * Not a general guard, and deliberately so: most `.warn` / `.hint` elements in this UI are static prose
 * that a screen reader already reads in document order, and a blanket "every warn needs aria-live" rule
 * would be noise with a false-positive rate that gets it disabled. This pins the small set of elements
 * that are THE ANSWER TO AN ACTION — rendered after a tap, out of the reading position the user was in.
 * Silent, they read as a button that did nothing.
 *
 * The e-stop verdict is the reason this file exists. An operator fires the stop, and "N of M peers NOT
 * confirmed stopped" — the single most urgent sentence this dashboard can produce — appeared with no live
 * region at all: seen, never heard. It is role=alert now; the all-clear is polite role=status, because
 * interrupting for good news trains people to ignore the channel.
 *
 * Run: node scripts/check-live-regions.mjs
 */
import fs from 'node:fs'
import path from 'node:path'

const SRC = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'src')
const PINNED = [
  { file: 'components/EstopSheet.tsx', what: 'the e-stop failure headline',
    find: /className="result bad"[^>]*role="alert"/ },
  { file: 'components/EstopSheet.tsx', what: 'the e-stop verdict (stopped / NOT confirmed)',
    find: /result\.all_stopped \? 'result ok' : 'result bad'[\s\S]{0,120}?role=\{result\.all_stopped \? 'status' : 'alert'\}/ },
  { file: 'components/RecordPanel.tsx', what: 'the record refusal / error line',
    find: /className="train-msg" role="alert"/ },
  { file: 'components/RunForm.tsx', what: "the policy-fit verdict when it BLOCKS the run",
    find: /role=\{fit\.blocking \? 'alert' : undefined\}/ },
]

const missing = []
for (const p of PINNED) {
  const src = fs.readFileSync(path.join(SRC, p.file), 'utf8')
  if (!p.find.test(src)) missing.push(p)
}
if (missing.length) {
  console.error(`FAIL  ${missing.length} answer(s) an operator waits for would arrive SILENTLY:`)
  for (const m of missing) console.error(`  - ${m.what}  (${m.file}) — no live-region role`)
  console.error('  These are rendered AFTER a tap, away from where the reader was: give the urgent ones '
    + 'role="alert" and the reassuring ones role="status". If the element moved, update this pin.')
  process.exit(1)
}
console.log(`live regions: ${PINNED.length} pinned answer(s) still announce themselves`)
