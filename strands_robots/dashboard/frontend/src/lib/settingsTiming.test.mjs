// Q51: npx esbuild src/lib/settingsMeta.ts --bundle --format=esm --outfile=/tmp/sm.mjs && node src/lib/settingsTiming.test.mjs
import assert from 'node:assert/strict'
const m = await import('/tmp/sm.mjs')
const { SETTINGS, APPLY_LABEL } = m

const byKey = k => SETTINGS.find(s => s.key === k)

// The camera rate is read by each ROBOT at its own start, so the honest chip is a respawn —
// "needs a mesh restart" pointed at a button that could only ever appear to work.
assert.equal(byKey('mesh.camera_hz').apply, 'respawn')
assert.match(APPLY_LABEL.respawn, /spawned from now on/)
assert.match(byKey('mesh.camera_hz').effect, /respawn/, 'the effect line must name the remedy')

// The keys the dashboard's own session reads keep their claim: over-correcting would hide a
// real restart requirement.
for (const k of ['mesh.port', 'mesh.connect', 'mesh.listen']) {
  assert.equal(byKey(k).apply, 'mesh-restart', `${k} really does need one`)
}

// Every mode has a label, and no label is empty — the chip is the whole claim.
for (const s of SETTINGS) {
  assert.ok(APPLY_LABEL[s.apply], `${s.key} has an unlabelled apply mode: ${s.apply}`)
  assert.ok(APPLY_LABEL[s.apply].length > 8, `${s.key}'s label says too little`)
}

console.log('settingsTiming: ok — ' + SETTINGS.length + ' tunables, every apply mode labelled')
