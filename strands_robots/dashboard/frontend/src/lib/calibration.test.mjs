// node assertions over the bundled module (esbuild first — see the sibling
// calibrateCommand.test.mjs for the same pattern).
//
// THE FIXTURE IS REAL: this is exactly what `GET /api/calibration` returned from the running
// dashboard on 2026-08-21, /Users/cagatay/.cache/huggingface/lerobot/calibration and all. An invented
// fixture would have been tidy, and tidy is how `None.json` went unnoticed for two days.
import assert from 'node:assert/strict'
import { parseCalibrationList, parseCalibrationDetail, idProblem } from '/tmp/calibration.mjs'

const LIVE = "**LeRobot Calibrations**\nLocation: `/Users/cagatay/.cache/huggingface/lerobot/calibration`\n\n## **Teleoperators**\n### **so101_leader** (2 calibrations)\n  - `leader` *(2025-11-23 22:18:06, 0.9KB, 6 motors)*\n  - `leader_arm` *(2025-11-17 22:47:29, 0.9KB, 6 motors)*\n\n### **so_leader** (1 calibrations)\n  - `None` *(2026-08-19 11:33:22, 0.9KB, 6 motors)*\n\n## **Robots**\n### **so101_follower** (3 calibrations)\n  - `follower` *(2025-11-23 22:17:33, 0.9KB, 6 motors)*\n  - `follower_arm` *(2025-11-17 22:42:39, 0.9KB, 6 motors)*\n  - `leader_arm` *(2025-11-17 22:40:09, 0.9KB, 6 motors)*\n\n### **so100_follower** (1 calibrations)\n  - `follower_arm` *(2025-11-17 23:10:20, 0.9KB, 6 motors)*\n\n### **so_follower** (3 calibrations)\n  - `follower` *(2026-08-15 22:36:47, 0.9KB, 6 motors)*\n  - `follower_arm` *(2026-08-15 22:36:47, 0.9KB, 6 motors)*\n  - `leader_arm` *(2025-11-17 22:40:09, 0.9KB, 6 motors)*\n"

const list = parseCalibrationList(LIVE)
assert.equal(list.location, '/Users/cagatay/.cache/huggingface/lerobot/calibration')
assert.equal(list.entries.length, 10, 'every id on this disk is listed')

// the device type comes from the ## heading and is lowercased for the API's device_type param
const byId = id => list.entries.filter(e => e.id === id)
assert.deepEqual(
  list.entries.filter(e => e.deviceType === 'teleoperators').map(e => `${e.model}/${e.id}`),
  ['so101_leader/leader', 'so101_leader/leader_arm', 'so_leader/None'],
)
assert.equal(byId('follower')[0].deviceType, 'robots')
assert.equal(byId('follower')[0].model, 'so101_follower')
assert.equal(byId('follower')[0].motors, 6)
assert.equal(byId('follower')[0].sizeKb, 0.9)
assert.equal(byId('follower')[0].modified, '2025-11-23 22:17:33')
assert.equal(byId('follower')[0].unreadable, false)
// the SAME id under two models is two different files and must stay two rows
assert.equal(byId('leader_arm').length, 3, 'leader_arm exists under three models')

// ── the real find: a python None that reached a file name ──
const none = byId('None')[0]
assert.ok(none, 'so_leader/None is on this disk')
assert.match(none.problem, /missing value that reached a file name/)
assert.match(none.problem, /Recalibrate under a real id/, 'the row says what to DO about it')
assert.equal(byId('follower')[0].problem, undefined, 'a normal id carries no problem')

// idProblem, on its own terms: every language's way of stringifying nothing
for (const bad of ['None', 'none', 'null', 'NULL', 'undefined', 'NaN', '  None  ', '']) {
  assert.ok(idProblem(bad), `refused as an id: "${bad}"`)
}
for (const ok of ['follower', 'leader_arm', 'none_of_your_business', 'arm-None-2', 'nonentity']) {
  assert.equal(idProblem(ok), undefined, `a real id is not touched: "${ok}"`)
}

// an unreadable file is a DIFFERENT verdict from a bad id, and still gets a row
const unreadable = parseCalibrationList('## **Robots**\n### **so101_follower**\n  - `broken` *(error reading file)*')
assert.equal(unreadable.entries.length, 1)
assert.equal(unreadable.entries[0].unreadable, true)
assert.equal(unreadable.entries[0].modified, undefined, 'no metadata is invented for a file that could not be read')

// empty and junk input parse to nothing rather than throwing
assert.deepEqual(parseCalibrationList('').entries, [])
assert.deepEqual(parseCalibrationList('nothing markdown about this').entries, [])

// ── the view action's per-motor table ──
const detail = parseCalibrationDetail([
  '**Calibration Details: `robots/so101_follower/follower`**',
  '**Path:** `/Users/cagatay/.cache/huggingface/lerobot/calibration/robots/so101_follower/follower.json`',
  '**Modified:** 2025-11-23 22:17:33',
  '**Size:** 921 bytes (0.9 KB)',
  '',
  '**Motor Configuration** (6 motors)',
  '',
  '### **shoulder_pan**',
  '  - **ID:** 1',
  '  - **Drive Mode:** 0',
  '  - **Homing Offset:** -2048',
  '  - **Range:** 700 to 3400',
  '### **wrist_roll**',
  '  - **ID:** 5',
  '  - **Range:** 950',
].join('\n'))
assert.equal(detail.title, 'robots/so101_follower/follower')
assert.match(detail.path, /follower\.json$/)
assert.equal(detail.size, '921 bytes (0.9 KB)')
assert.equal(detail.motors.length, 2)
assert.deepEqual(detail.motors[0], { name: 'shoulder_pan', id: '1', driveMode: '0', homingOffset: '-2048', rangeMin: '700', rangeMax: '3400' })
// a range with no "to" keeps the value it HAS rather than inventing a max: a fabricated joint limit
// is the one kind of wrong answer this table must never give.
assert.equal(detail.motors[1].rangeMin, '950')
assert.equal(detail.motors[1].rangeMax, undefined)
// fields before any motor heading belong to nobody and are dropped, not attached to motor 1
assert.deepEqual(parseCalibrationDetail('  - **ID:** 9').motors, [])

console.log('calibration: all assertions passed')
