import assert from 'node:assert/strict'
import { boardListEmptyLine, managedListEmptyLine, cameraGridEmptyLine } from '/tmp/boardList.mjs'

// --- the defect: a failed request rendered as a hardware verdict ---------------
{
  const v = boardListEmptyLine({ scanned: false, error: 'HTTP 401' })
  assert.equal(v.kind, 'unscanned')
  assert.match(v.message, /scan failed \(HTTP 401\)/)
  assert.match(v.message, /not because nothing is plugged in/, 'name the ambiguity, do not resolve it')
  assert.doesNotMatch(v.message, /no servo board detected/, 'a failed scan may never claim absent hardware')
}

// --- nothing asked yet: also not a verdict ------------------------------------
{
  const v = boardListEmptyLine({ scanned: false })
  assert.equal(v.kind, 'scanning')
  assert.match(v.message, /scanning USB/)
  assert.doesNotMatch(v.message, /no servo board/)
  // null and '' are the same as absent — an error state cleared to null must not
  // read as an error string.
  assert.equal(boardListEmptyLine({ scanned: false, error: null }).kind, 'scanning')
  assert.equal(boardListEmptyLine({ scanned: false, error: '  ' }).kind, 'scanning')
}

// --- the one case that MAY talk about hardware --------------------------------
{
  const v = boardListEmptyLine({ scanned: true })
  assert.equal(v.kind, 'detected')
  assert.match(v.message, /no servo board detected/)
  assert.match(v.message, /nothing on USB enumerated as a serial bus/, 'say what was looked for')
}

// --- a stale error alongside a successful scan is kept, not hidden ------------
{
  const v = boardListEmptyLine({ scanned: true, error: 'refresh timed out' })
  assert.equal(v.kind, 'detected')
  assert.match(v.message, /no servo board detected/)
  assert.match(v.message, /last refresh also reported: refresh timed out/)
}

// --- managed robots: a failed scan must not report zero children ---------------
{
  const failed = managedListEmptyLine({ scanned: false, error: 'HTTP 401' })
  assert.equal(failed.kind, 'unscanned')
  assert.match(failed.message, /nothing answered, not because there are none/)
  assert.doesNotMatch(failed.message, /None\./, 'children may be running, publishing and holding ports')

  assert.equal(managedListEmptyLine({ scanned: false }).kind, 'scanning')

  const answered = managedListEmptyLine({ scanned: true })
  assert.equal(answered.kind, 'detected')
  assert.match(answered.message, /^None\./, 'only an answered scan may say none')
  assert.match(answered.message, /joins the mesh as its own peer/, 'keep the explanation that was there')
}

// --- cameras: do not send someone to the cable from an unanswered request -------
{
  const failed = cameraGridEmptyLine({ scanned: false, error: 'Failed to fetch' })
  assert.equal(failed.kind, 'unscanned')
  assert.doesNotMatch(failed.message, /plug one in/, 'an unanswered scan cannot ask for hardware')
  assert.match(failed.message, /nothing answered, not because there are no cameras/)

  assert.equal(cameraGridEmptyLine({ scanned: false }).kind, 'scanning')

  const answered = cameraGridEmptyLine({ scanned: true })
  assert.equal(answered.kind, 'detected')
  // "No cameras probed" claimed the probe did not happen; it did, and nothing answered.
  assert.match(answered.message, /No camera index answered a probe/)
  assert.match(answered.message, /rescan if you just did/)
}

// --- one wording for "nothing answered", across every list on the screen --------
{
  const a = boardListEmptyLine({ scanned: false, error: 'HTTP 401' }).message
  const b = managedListEmptyLine({ scanned: false, error: 'HTTP 401' }).message
  const c = cameraGridEmptyLine({ scanned: false, error: 'HTTP 401' }).message
  for (const m of [a, b, c]) assert.match(m, /the device scan failed \(HTTP 401\) — this list is empty because nothing answered/)
}

console.log('boardList: all assertions passed')
