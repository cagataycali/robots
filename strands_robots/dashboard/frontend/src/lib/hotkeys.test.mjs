import assert from 'node:assert/strict'
import { hotkeyVerdict, isTyping, ESTOP_KEYSHORTCUTS } from '/tmp/hotkeys.mjs'

const ev = (o) => ({ key: '', ...o })

// --- the brake works from anywhere, which is the point of JOURNEYS #12 -------
assert.equal(hotkeyVerdict(ev({ key: '.' })), 'estop')
assert.equal(hotkeyVerdict(ev({ key: '.', metaKey: true })), 'estop')
assert.equal(hotkeyVerdict(ev({ key: '.', ctrlKey: true })), 'estop')
// ...INCLUDING mid-sentence in the record/run form, the worst case the audit found
for (const tag of ['INPUT', 'TEXTAREA', 'SELECT', 'input', 'textarea']) {
  assert.equal(hotkeyVerdict(ev({ key: '.', metaKey: true, targetTag: tag })), 'estop', tag)
  assert.equal(hotkeyVerdict(ev({ key: '.', ctrlKey: true, targetTag: tag })), 'estop', tag)
}
assert.equal(hotkeyVerdict(ev({ key: '.', metaKey: true, editable: true })), 'estop')

// --- but a typed full stop must never fire it ------------------------------- A task sentence
// ends in ".".
assert.equal(hotkeyVerdict(ev({ key: '.', targetTag: 'INPUT' })), null)
assert.equal(hotkeyVerdict(ev({ key: '.', targetTag: 'TEXTAREA' })), null)
assert.equal(hotkeyVerdict(ev({ key: '.', editable: true })), null)

// Alt is excluded: Alt+. produces characters on several layouts, so treating it
// as a chord would hijack real typing.
assert.equal(hotkeyVerdict(ev({ key: '.', altKey: true })), null)
assert.equal(hotkeyVerdict(ev({ key: '.', altKey: true, metaKey: true })), null)

// --- help ------------------------------------------------------------------
assert.equal(hotkeyVerdict(ev({ key: '?' })), 'help')
assert.equal(hotkeyVerdict(ev({ key: '?', targetTag: 'INPUT' })), null, '"?" is a character people type')
assert.equal(hotkeyVerdict(ev({ key: '?', metaKey: true })), null, 'no chord is advertised for help')

// --- escape belongs to everyone --------------------------------------------
for (const t of [undefined, 'INPUT', 'TEXTAREA', 'DIV']) {
  assert.equal(hotkeyVerdict(ev({ key: 'Escape', targetTag: t })), 'close')
}
assert.equal(hotkeyVerdict(ev({ key: 'Escape', metaKey: true })), 'close')

// --- ordinary keys stay ordinary -------------------------------------------
for (const k of ['a', 'Enter', 'Tab', ' ', 'ArrowDown', 'Shift', ',', '/']) {
  assert.equal(hotkeyVerdict(ev({ key: k })), null, k)
}
// a chord that is not the stop chord must not stop the fleet
assert.equal(hotkeyVerdict(ev({ key: 'k', metaKey: true })), null)
assert.equal(hotkeyVerdict(ev({ key: 'r', ctrlKey: true })), null, 'Ctrl+R is reload, not a stop')

assert.equal(isTyping(ev({ key: 'a', targetTag: 'INPUT' })), true)
assert.equal(isTyping(ev({ key: 'a', targetTag: 'DIV' })), false)

// what the button promises must be what the handler honours
assert.equal(ESTOP_KEYSHORTCUTS, '. Meta+. Control+.')
for (const chord of ESTOP_KEYSHORTCUTS.split(' ')) {
  const e = ev({ key: '.', metaKey: /Meta/.test(chord), ctrlKey: /Control/.test(chord) })
  assert.equal(hotkeyVerdict(e), 'estop', `advertised ${chord} must work`)
}

console.log('hotkeys: all assertions passed')
