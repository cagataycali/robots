// Assertions for the joint history ring + its stall test (lib/jointHistory.ts).
// Run: npx esbuild src/lib/jointHistory.ts --bundle --format=esm --outfile=/tmp/jointHistory.mjs \
//        && node src/lib/jointHistory.test.mjs
import assert from 'node:assert/strict'

const { createHistory, pushFrame, traceFor, stalled, GAP_MS } = await import('/tmp/jointHistory.mjs')

// The window's right edge is NOW: that is the property the sparkline's honesty
// rests on, and the reason a frozen canvas is a lie rather than a stale picture.
{
  const h = createHistory()
  pushFrame(h, [['shoulder_pan', 10]], 1_000)
  pushFrame(h, [['shoulder_pan', 20]], 2_000)
  const pts = traceFor(h.get('shoulder_pan'), 2_000, { lo: 0, hi: 100 }, 100, 20)
  assert.equal(pts.length, 2)
  assert.equal(Math.round(pts[1].x), 100, 'newest sample sits at the right edge')
  assert.ok(pts[0].x < pts[1].x, 'older is left of newer')
  // Ten seconds later with no new frame, the same data must have MOVED LEFT.
  const later = traceFor(h.get('shoulder_pan'), 12_000, { lo: 0, hi: 100 }, 100, 20)
  assert.ok(later[later.length - 1].x < 90, 'a stalled trace slides away from now')
}

// A hole longer than GAP_MS breaks the line instead of inventing motion.
{
  const h = createHistory()
  pushFrame(h, [['j', 1]], 1_000)
  pushFrame(h, [['j', 2]], 1_000 + GAP_MS + 50)
  const pts = traceFor(h.get('j'), 1_000 + GAP_MS + 50, { lo: 0, hi: 10 }, 100, 20)
  assert.equal(pts[0].gapAfter, true)
}

// stalled(): drives the sparkline's redraw decision under prefers-reduced-motion,
// where the ticker used to be off entirely and a dead stream drew itself as a
// still arm (x is TIME with now at the right edge).
{
  const now = 10_000
  assert.equal(stalled(undefined, now), false, 'nothing received is not a stall')
  assert.equal(stalled([], now), false)
  assert.equal(stalled([{ t: now - 100, v: 1 }], now), false, 'a live 10Hz stream is not a stall')
  assert.equal(stalled([{ t: now - 799, v: 1 }], now), false)
  assert.equal(stalled([{ t: now - 801, v: 1 }], now), true)
  assert.equal(stalled([{ t: now - 60_000, v: 1 }], now), true)
  // It reads the NEWEST sample, not the oldest.
  assert.equal(stalled([{ t: now - 60_000, v: 1 }, { t: now - 50, v: 2 }], now), false)
  // Explicit gap override, so the caller owns the policy.
  assert.equal(stalled([{ t: now - 300, v: 1 }], now, 200), true)
}
console.log('jointHistory: stalled() assertions passed')
