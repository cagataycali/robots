// A CLAMP HIDES TEXT. This pairs the two halves that make that safe (Q115): the grid card clamps
// the joint-absence remedy to 4 lines, and JointStrip puts the WHOLE remedy in the element's
// title so nothing is lost. Removing either half alone is silent — the card still looks fine,
// and the sentence that says "Do NOT recalibrate" just stops being readable. CSS cannot be unit
// tested here, so this is the only witness.
import fs from 'node:fs'
const css = fs.readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8')
const tsx = fs.readFileSync(new URL('../src/components/JointStrip.tsx', import.meta.url), 'utf8')
const clamped = /\.card\s+\.joints\.empty\s+\.hint\s*\{[^}]*line-clamp/.test(css)
const titled = /className="hint"[^>]*title=/.test(tsx)
if (clamped !== titled) {
  console.error(clamped
    ? 'FAIL: the card clamps the remedy but JointStrip no longer sets title — the clamped text is now unreachable.'
    : 'FAIL: JointStrip sets title for a clamp that no longer exists — either restore the clamp or drop the title.')
  process.exit(1)
}
console.log(`ok clamp+title pairing (${clamped ? 'both present' : 'both absent'})`)
