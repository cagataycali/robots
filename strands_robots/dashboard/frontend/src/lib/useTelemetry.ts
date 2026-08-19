import { useEffect, useRef, useState } from 'react'
import type { Peer } from '../types'

const CAP = 120 // ~12 s at the 10 Hz state topic

export interface TelemetrySample { t: number; motion: number }

export interface Telemetry {
  samples: TelemetrySample[]
  /** measured state-topic rate, Hz (0 until 2+ samples) */
  hz: number
  /** joints changed recently; null until enough samples to judge */
  moving: boolean | null
  /** the peer publishes joint positions at all; null before any state sample */
  jointsSeen: boolean | null
  /** seconds since the newest state sample; null before the first one */
  stateAgeS: number | null
}

function jointValues(peer: Peer): number[] {
  const joints = peer.state?.joints
  if (!joints) return []
  return Object.values(joints).map(v => {
    if (typeof v === 'number') return v
    if (Array.isArray(v)) return v[0] ?? 0
    return (v as { position?: number }).position ?? 0
  })
}

/**
 * The state-topic telemetry ring, as a hook.
 *
 * Extracted from TelemetryStrip so the robot card's status sentence and the
 * sparkline read the SAME motion judgment - two independent motion detectors
 * on one card would eventually disagree in front of the user, and a card
 * that says "still" next to a wiggling sparkline teaches them to trust
 * neither.
 *
 * `moving` is null (no opinion) until at least 10 samples exist: the status
 * sentence uses motion to accuse a policy of being wedged or to warn about
 * an uncommanded arm, and an accusation off 2 samples is noise.
 */
export function useTelemetry(peer: Peer): Telemetry {
  const ring = useRef<TelemetrySample[]>([])
  const prev = useRef<number[]>([])
  // null until the first state sample arrives: "no joints yet" and "never any
  // joints" must not look the same before we have heard anything at all.
  const jointsPresent = useRef<boolean | null>(null)
  const lastT = useRef<number | undefined>(undefined)
  const [, tick] = useState(0)

  const stateT = peer.state?.t

  useEffect(() => {
    if (stateT === undefined || stateT === lastT.current) return
    lastT.current = stateT
    const values = jointValues(peer)
    jointsPresent.current = (jointsPresent.current ?? false) || values.length > 0
    let motion = 0
    if (prev.current.length === values.length && values.length) {
      for (let i = 0; i < values.length; i++) motion += Math.abs(values[i] - prev.current[i])
      motion /= values.length
    }
    prev.current = values
    ring.current = [...ring.current, { t: Date.now() / 1000, motion }].slice(-CAP)
    tick(n => n + 1)
  }, [stateT]) // eslint-disable-line react-hooks/exhaustive-deps

  const samples = ring.current
  // Did any sample carry joint positions? A peer that publishes none can never
  // produce a motion measurement, and the status sentence must say so rather
  // than reporting the resulting silence as stillness.
  const jointsSeen = jointsPresent.current
  if (samples.length < 2) return { samples, hz: 0, moving: null, stateAgeS: null, jointsSeen }

  const span = samples[samples.length - 1].t - samples[0].t
  const hz = span > 0 ? (samples.length - 1) / span : 0
  const peak = Math.max(...samples.map(s => s.motion), 1e-6)
  // NO JOINTS MEANS NO OPINION. `motion` is computed from joint positions, so a
  // peer that publishes none yields motion 0 on every sample - and after 10
  // samples this returned `false`, i.e. a MEASURED stillness manufactured out of
  // an empty stream. That is how a card with "no joint data on this peer" came to
  // display "idle and still - safe to approach" (live, so101-arm-1).
  const moving = jointsSeen && samples.length >= 10
    ? samples.slice(-10).some(s => s.motion > peak * 0.05)
    : null
  const stateAgeS = Date.now() / 1000 - samples[samples.length - 1].t
  return { samples, hz, moving, stateAgeS, jointsSeen }
}

export const TELEMETRY_CAP = CAP
