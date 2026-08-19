import { useEffect, useRef } from 'react'
import type { Range } from '../lib/jointScale'
import { traceFor, type Sample, HISTORY_WINDOW_MS } from '../lib/jointHistory'

/**
 * One joint's recent past, drawn on a canvas (U6).
 *
 * Canvas rather than SVG because there is one of these per joint per card and
 * they redraw at stream rate: 6 arms x 6 joints of DOM polyline with 900 points
 * each is how a dashboard starts dropping frames while claiming to be realtime.
 *
 * The redraw is driven by `frame` (a counter the parent bumps when new state
 * arrives) and by a slow ticker, so the trace keeps sliding left even when the
 * robot is idle - a frozen sparkline and a still arm must not look alike.
 */
export default function JointSpark({
  track,
  range,
  frame,
  height = 22,
  live = true,
}: {
  track?: Sample[]
  range: Range
  frame: number
  height?: number
  live?: boolean
}) {
  const ref = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    let raf = 0
    let timer: number | undefined

    const draw = () => {
      const parent = canvas.parentElement
      const cssW = Math.max(1, parent?.clientWidth ?? canvas.clientWidth)
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(cssW * dpr)
        canvas.height = Math.round(height * dpr)
        canvas.style.width = `${cssW}px`
        canvas.style.height = `${height}px`
      }
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, cssW, height)

      const now = Date.now()
      const pts = traceFor(track, now, range, cssW, height)

      // mid-line: the middle of the joint's own range, so a trace hugging it
      // means "near the centre of travel" and not "near zero" by coincidence
      const styles = getComputedStyle(canvas)
      ctx.strokeStyle = styles.getPropertyValue('--spark-axis').trim() || 'rgba(255,255,255,.08)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, Math.round(height / 2) + 0.5)
      ctx.lineTo(cssW, Math.round(height / 2) + 0.5)
      ctx.stroke()

      if (pts.length === 0) return

      const line = styles.getPropertyValue('--spark-line').trim() || '#4a9eff'
      // a soft fill under the trace gives depth without a second colour
      const fill = ctx.createLinearGradient(0, 0, 0, height)
      fill.addColorStop(0, styles.getPropertyValue('--spark-fill').trim() || 'rgba(74,158,255,.22)')
      fill.addColorStop(1, 'rgba(0,0,0,0)')

      let i = 0
      while (i < pts.length) {
        let j = i
        while (j < pts.length - 1 && !pts[j].gapAfter) j++
        const seg = pts.slice(i, j + 1)
        if (seg.length > 1) {
          ctx.beginPath()
          ctx.moveTo(seg[0].x, seg[0].y)
          for (const p of seg.slice(1)) ctx.lineTo(p.x, p.y)
          ctx.strokeStyle = line
          ctx.lineWidth = 1.4
          ctx.lineJoin = 'round'
          ctx.stroke()
          ctx.lineTo(seg[seg.length - 1].x, height)
          ctx.lineTo(seg[0].x, height)
          ctx.closePath()
          ctx.fillStyle = fill
          ctx.fill()
        }
        i = j + 1
      }

      // the head: where the joint is right now
      const head = pts[pts.length - 1]
      ctx.beginPath()
      ctx.arc(head.x, head.y, 1.9, 0, Math.PI * 2)
      ctx.fillStyle = line
      ctx.fill()
    }

    raf = requestAnimationFrame(draw)
    // Idle robots still need the window to scroll; 4Hz is invisible work.
    // Under prefers-reduced-motion the trace only advances when new state
    // arrives: the data is still complete, it just stops crawling on its own.
    const calm = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (live && !calm) timer = window.setInterval(() => { raf = requestAnimationFrame(draw) }, 250)
    return () => {
      cancelAnimationFrame(raf)
      if (timer) window.clearInterval(timer)
    }
  }, [track, range.lo, range.hi, frame, height, live])

  return (
    <canvas
      ref={ref}
      className="jspark"
      role="img"
      aria-label={`last ${Math.round(HISTORY_WINDOW_MS / 1000)} seconds of movement`}
    />
  )
}
