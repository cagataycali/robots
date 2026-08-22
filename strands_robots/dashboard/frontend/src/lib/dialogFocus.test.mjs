// Run: node scripts/run-lib-tests.mjs dialogFocus Focus bugs are the quietest bugs in the app:
// nothing throws, nothing looks wrong in a screenshot, and the only person who notices is the
// one who cannot use a mouse.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const { focusPlan, looksLikeClose, looksDangerous, rememberOpener, shouldRestoreFocus } =
  await import('/tmp/dialogFocus.mjs')

for (const label of ['close the activity log', 'close devices', 'Close help', 'close', 'close settings',
                     'close training', 'close this robot', 'dismiss this hint']) {
  assert.ok(looksLikeClose({ label }), `"${label}" is a close affordance`)
}
assert.ok(looksLikeClose({ text: '✕' }), 'a bare glyph button has no words at all, and is always close here')
assert.ok(looksLikeClose({ text: '×' }), 'the other glyph')
assert.ok(!looksLikeClose({ label: 'closest camera' }), 'a word STARTING with close is not the word close')
assert.ok(!looksLikeClose({ label: 'disclose the token' }), 'nor one containing it')
assert.ok(!looksLikeClose({ text: 'x axis' }), 'a lone x is a glyph; "x axis" is a label')

// ── 2. controls that must never be auto-focused ── Focus is not activation — but a keyboard
// user who lands on `▶ run` and presses space has commanded a real arm.
for (const text of ['▶ run', 'start recording', '⏺ record', 'stop', 'E-STOP', 'delete episode',
                    'despawn', 'deploy to arm-1', 'train', 'calibrate', 'replay in sim', 'teleop']) {
  assert.ok(looksDangerous({ text }), `"${text}" moves hardware or destroys work`)
}
for (const text of ['episodes', 'settings', 'dataset name', 'cancel', 'refresh', 'back']) {
  assert.ok(!looksDangerous({ text }), `"${text}" is safe to focus`)
}

// ── 3. the plan ──
assert.equal(focusPlan([{ text: 'run' }, { autofocus: true, text: 'name' }]), 1,
             'the overlay saying where to start wins over everything')
assert.equal(focusPlan([{ text: 'run' }, { label: 'close settings' }, { text: 'name' }]), 1,
             'then the close button — the safe default, like HelpSheet always did')
assert.equal(focusPlan([{ text: 'episodes' }, { text: 'run' }]), 0, 'then the first SAFE control')

assert.equal(focusPlan([]), 'container',
             'an empty sheet focuses the DIALOG, which is a real answer: a screen reader lands inside ' +
             'it and Tab starts inside it')

// ...and the same answer when everything present is dangerous. Better a container than a run button.
assert.equal(focusPlan([{ text: '▶ run' }, { text: 'stop' }]), 'container',
             'a sheet offering only hardware controls gets the container, never the first of them')
assert.equal(focusPlan([{ text: '▶ run' }, { label: 'close', text: '✕' }]), 1,
             'but a close button among them is still preferred over the container')

// ── 4.
const body = { tag: 'body' }
const chip = { tag: 'button' }
assert.deepEqual(rememberOpener(chip, body), { el: chip }, 'a real element is the opener')
assert.equal(rememberOpener(body, body), null, 'Q92: body is NOT an opener — leave focus alone on close')
assert.equal(rememberOpener(null, body), null, 'nor is nothing')

// ── 5. giving focus back ──
assert.equal(shouldRestoreFocus({ activeInsideOverlay: true, activeIsBody: false, openerConnected: true }),
             true, 'the overlay still holds focus: hand it back')
assert.equal(shouldRestoreFocus({ activeInsideOverlay: false, activeIsBody: true, openerConnected: true }),
             true, 'focus fell to nowhere when the sheet unmounted: hand it back')
assert.equal(shouldRestoreFocus({ activeInsideOverlay: false, activeIsBody: false, openerConnected: true }),
             false, 'the operator has already clicked elsewhere — yanking their focus is worse')
assert.equal(shouldRestoreFocus({ activeInsideOverlay: true, activeIsBody: false, openerConnected: false }),
             false, 'the opener is gone from the document; focusing a detached node does nothing')

// ── 6. the wiring, because a pure table stays true of a function nobody calls ──
const src = readFileSync(new URL('./useDialogFocus.ts', import.meta.url), 'utf8')
assert.match(src, /const plan = focusPlan\(/, 'the hook must ASK for the plan')
assert.match(src, /if \(plan === 'container'\)[\s\S]{0,400}node\.focus\(\)/,
             'and act on container by focusing the dialog itself')
assert.match(src, /setAttribute\('tabindex', '-1'\)/, 'which needs tabindex to be focusable at all')
assert.match(src, /rememberOpener\(document\.activeElement, document\.body\)/, 'Q92 is applied, not just written')
assert.match(src, /shouldRestoreFocus\(\{/, 'and so is the restore rule')
assert.doesNotMatch(src, /querySelector<HTMLElement>\('\[data-autofocus\]'\)/,
                    'the old chained ?? must be gone, not shadowing the plan')

console.log('dialogFocus.test.mjs: all assertions passed')
